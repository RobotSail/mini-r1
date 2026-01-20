import sys
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data import DistributedSampler
import requests
from transformers import GenerationConfig
import json
import random
from typer import Typer
import typer
import re
import pydantic
import datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2Tokenizer,
    PreTrainedModel,
)
import torch
from torch.optim import AdamW
import os
from IPython import embed
from tqdm import tqdm
import torch.distributed as dist
import logging
from rich.logging import RichHandler

from src.data_utils import (
    generate_dataset,
    # dataset_from_groups,
    # create_grpo_data_loader,
    JsonlDataset,
    ProblemDataset,
    # create_distributed_data_loader,
    problem_collate_fn,
    collate_fn as grpo_collate_fn,
)
from src.distributed_packer import (
    create_distributed_grpo_dataloader,
    DistributedPackingSampler,
    PackedBatch,
    AccumulationWindow,
)
from utils import (
    display_scorecard,
    init_distributed,
    destroy_distributed_environment,
    log_rank_0,
)
from type_defs import (
    Problem,
    SamplingParams,
    TokenSample,
    RolloutResult,
    Sample,
    TrainingContext,
    Hyperparameters,
)
import threading
import sys
import subprocess
import torch.distributed.fsdp as fsdp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
    set_model_state_dict,
)
import wandb
from src.data_utils import create_oleg_grpo_dataloader

# # load the model once and save
from model_utils import prepare_model


# Regex pattern to match <answer>...</answer> tags
answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=False, show_path=False)],
)
log = logging.getLogger("mini-r1")
# httpx is verbose so we limit the verbosity here
logging.getLogger("httpx").setLevel(logging.WARNING)

app = Typer(pretty_exceptions_enable=False)


@app.command()
def generate_data(
    # system_msg: str,
    num_problems: int = 20,
    min_num: int = -100,
    max_num: int = 100,
    seed: int = 42,
    output_dir: str = "generated_data",
    test_split: float = 0.0,
    max_seq_len: int = 8192,
    # system_msg: str | None ="You are a helpful math assistant. Always provide your final numerical answer inside of the <answer>...</answer> tags, e.g.: <answer>42</answer>",
    system_msg: str | None = None,
):
    # this is the dataset
    dataset: datasets.Dataset = generate_dataset(
        system_msg=system_msg,
        seed=seed,
        num_problems=num_problems,
        min_num=min_num,
        max_num=max_num,
    )
    if test_split > 0:
        dataset_dict = dataset.train_test_split(test_split)
        train, test = dataset_dict["train"], dataset_dict["test"]
    else:
        train = dataset
        test = None
    os.makedirs(output_dir, exist_ok=True)

    # write out training data
    train_path = os.path.join(output_dir, "train.jsonl")
    train.to_json(train_path)
    log.info(f"✓ Generated {len(train)} training examples")
    log.info(f"✓ Saved training data to '{train_path}'")

    # write out test data if it exists
    if test:
        test_path = os.path.join(output_dir, "test.jsonl")
        test.to_json(test_path)
        log.info(f"✓ Generated {len(test)} test examples")
        log.info(f"✓ Saved test data to '{test_path}'")


@app.command()
def eval(
    eval_path: str = typer.Option(
        ..., "--eval-path", help="Path to the evaluation dataset (jsonl)"
    ),
    model_name: str = typer.Option(..., "--model", "-m", help="Model name or path"),
    gpu: int = typer.Option(0, "--gpu", "-g", help="CUDA GPU index to use"),
    max_new_tokens: int = typer.Option(
        128, help="Maximum number of new tokens to generate"
    ),
    max_seq_len: int = typer.Option(
        8192, "--msl", "--max-seq-len", help="Maximum sequence length"
    ),
    temperature: float = typer.Option(0.7, "-t", "--temp", help="Sampling temperature"),
    group_size: int = typer.Option(
        1, "-G", "--group-size", help="Number of rollouts per prompt (for pass@k)"
    ),
):
    """Run evaluation on a dataset without training."""
    device = torch.device("cuda", gpu)

    eval_dataset = datasets.load_dataset("json", data_files=eval_path, split="train")
    log.info(f"✓ Loaded {len(eval_dataset)} evaluation samples")

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id and not model.config.pad_token_id:
        model.config.pad_token_id = tokenizer.pad_token_id

    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        max_tokens=max_seq_len,
        top_p=1.0,
        top_k=0.0,
        repetition_penalty=1.0,
    )

    eval_data = eval_dataset.batch(eval_dataset.num_rows)

    percent_scores = []
    for sample in samples:
        passing_rate = sum(1 if r.is_correct else 0 for r in sample.rollouts) / len(
            sample.rollouts
        )
        percent_scores.append(passing_rate)

    percent_above_50 = (
        sum(1 if score > 0.5 else 0 for score in percent_scores)
        / len(percent_scores)
        * 100
    )
    percent_at_100 = (
        sum(1 if score == 1.0 else 0 for score in percent_scores)
        / len(percent_scores)
        * 100
    )

    log.info("=== Evaluation Results ===")
    log.info(f"Model: {model_name}")
    log.info(f"Samples: {len(samples)}")
    log.info(
        f"Pass@{group_size}: {percent_above_50:.1f}% above 50% | {percent_at_100:.1f}% at 100%"
    )


class GRPOTrainer:
    """
    GRPO Training class for mini R1.

    TODO: move components from the TrainingContext object into here
    """

    ctx: TrainingContext
    train_dataset: ProblemDataset
    eval_dataset: ProblemDataset
    train_loader: torch.utils.data.DataLoader

    def __init__(self, ctx: TrainingContext):
        self.ctx = ctx
        # load the dataset
        if ctx.dataset == "json":
            self.train_dataset, self.eval_dataset = ProblemDataset.from_jsonl(
                ctx.train_path, ctx.eval_split
            )
        elif ctx.dataset == "gsm8k":
            self.train_dataset, self.eval_dataset = ProblemDataset.from_gsm8k(
                ctx.eval_split
            )
        self._training_step = 0

        # Print dataset statistics
        log.info(f"✓ Loaded {len(self.train_dataset)} training samples")
        if self.eval_dataset:
            log.info(f"✓ Loaded {len(self.eval_dataset)} evaluation samples")

        # check if we need to write into output dir
        if self.ctx.output_dir is not None and not self.ctx.valid_save_dir():
            log.error(f"Cannot write to output directory '{self.ctx.output_dir}'")
            raise typer.Exit(code=1)

    def initialize(self):
        # load models
        self.ctx.load_models()
        self.ctx.create_device_mesh()  # Must be called before FSDP wrapping
        self.ctx.wrap_models_with_fsdp2()

        # Create the train loader now that distributed is initialized
        self.train_loader = self._create_problem_data_loader(
            self.train_dataset, self.ctx.hparams.batch_size
        )

    def _create_problem_data_loader(self, dataset: ProblemDataset, batch_size: int):
        """Create a distributed data loader for the problem dataset (for rollout generation)."""
        train_sampler = DistributedSampler(
            dataset=dataset,
            drop_last=True,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            seed=67,
            shuffle=True,
        )
        batchsize = batch_size // dist.get_world_size()
        return DataLoader(
            dataset,
            sampler=train_sampler,
            batch_size=batchsize,
            collate_fn=problem_collate_fn,
        )

    @torch.no_grad
    def simple_rollouts(
        self, batch: list[Problem], max_seq_len: int, is_eval=False, n_completions=None
    ):
        log_rank_0("making call to update vLLM policy")
        self.ctx.update_vllm_policy()
        if not is_eval:
            return self.ctx.generate_completions(batch, max_seq_len=max_seq_len)

        kwargs = {}
        if n_completions is not None and n_completions > 0:
            kwargs["n_completions"] = n_completions
        return self.ctx.generate_completions(
            batch,
            max_seq_len=max_seq_len,
            include_logprobs=False,
            only_run_on_main=True,
            **kwargs,
        )

    @torch.no_grad
    def eval_model(self):
        # here we want to evaluate the model with vllm
        if not self.eval_dataset or len(self.eval_dataset) == 0:
            log_rank_0("no eval dataset present, skipping evaluation")
            return

        # we generate all the rollouts
        # eval_data = eval_dataset.batch(eval_dataset.num_rows)
        pass_at = [
            1,
        ]  #  3,#  5, 10]
        results = []
        eval_problems = [prob for prob in self.eval_dataset]

        for npass in pass_at:
            rollout_msl = (
                self.ctx.hparams.msl_pre
                if self._training_step < self.ctx.hparams.msl_jump_at_step
                else self.ctx.hparams.msl_post
            )
            # need to generate the whole dataset
            samples = self.simple_rollouts(
                eval_problems,
                max_seq_len=rollout_msl,
                is_eval=True,
                n_completions=npass,
            )

            # now we go and determine the passing rate
            percent_scores = []
            for sample in samples:
                passing_rate = sum(
                    1 if r.score.is_correct else 0 for r in sample.rollouts
                ) / len(sample.rollouts)
                percent_scores.append(passing_rate)
            # Calculate statistics
            percent_above_50 = (
                sum(1 if score > 0.5 else 0 for score in percent_scores)
                / max(len(percent_scores), 1)
                * 100
            )
            percent_at_100 = (
                sum(1 if score == 1.0 else 0 for score in percent_scores)
                / max(len(percent_scores), 1)
                * 100
            )

            results.append((npass, percent_above_50, percent_at_100))

        # Print all results at the end
        log.info("=== Evaluation Scorecard ===")
        log.info(f"Total samples evaluated: {len(samples)}")
        for npass, percent_above_50, percent_at_100 in results:
            log.info(
                f"Pass@{npass}: {percent_above_50:.1f}% above 50% | {percent_at_100:.1f}% at 100% (across {len(samples)} samples with {npass} rollout(s) each)"
            )

        # Log evaluation metrics to wandb
        if dist.get_rank() == 0:
            eval_metrics = {"eval/num_samples": len(samples)}
            for npass, percent_above_50, percent_at_100 in results:
                eval_metrics[f"eval/pass@{npass}_above_50"] = percent_above_50
                eval_metrics[f"eval/pass@{npass}_at_100"] = percent_at_100
            wandb.log(eval_metrics, step=self._training_step)

    def create_grpo_data_loader(
        self,
        samples: list[Sample],
    ):
        """
        Create a distributed data loader for GRPO training with lockstep microbatches.

        This uses FFD packing with K-synchronization to ensure all ranks have
        exactly the same number of microbatches, preventing FSDP collective mismatches.

        The dataloader groups microbatches into accumulation windows, where each window
        contains K microbatches to accumulate before taking an optimizer step.
        """
        from functools import partial

        # Create the collate function with pad token
        _collate_fn = partial(
            grpo_collate_fn, pad_token_id=self.ctx.tokenizer.pad_token_id
        )

        # Create the JsonlDataset from samples
        ds = JsonlDataset(dataset=samples, pad_token_id=self.ctx.tokenizer.pad_token_id)

        # Calculate accumulation steps: how many microbatches to process before optimizer.step()
        # We want to accumulate roughly inner_batch_size samples per optimizer step
        # Since we're using distributed training, divide by world size
        world_size = dist.get_world_size()
        accumulation_steps = max(1, self.ctx.hparams.inner_batch_size // world_size)
        accumulation_steps = None

        log_rank_0(
            f"Using {accumulation_steps} accumulation steps per optimizer update"
        )

        # Use the distributed packing dataloader
        # This ensures all ranks do exactly K microbatches per accumulation window
        return create_distributed_grpo_dataloader(
            dataset=ds,
            max_tokens_per_rank=self.ctx.hparams.max_tokens_per_gpu,
            device=self.ctx.device,
            collate_fn=_collate_fn,
            accumulation_steps=accumulation_steps,
            use_quadratic_cost=True,  # Better for attention-dominated workloads
            shuffle=False,  # Already randomized by rollout sampling
        )

    @torch.no_grad()
    def update_reference_policy(self):
        log_rank_0("updating reference policy")
        dist.barrier()

        # first we must gather original policy
        policy_sd = get_model_state_dict(
            self.ctx.model,
            options=StateDictOptions(
                ignore_frozen_params=True,
                full_state_dict=False,
            ),
        )
        set_model_state_dict(
            self.ctx.ref_model,
            policy_sd,
            options=StateDictOptions(ignore_frozen_params=False, full_state_dict=False),
        )
        dist.barrier()
        log_rank_0("finished updating state dicts")

    @torch.no_grad()
    def take_training_step(self):
        # take an optimization step
        # TODO: responsibility needs to be redistributed between ctx and trainer
        # this optimize step for example should probably live here
        gradnorm = self.ctx.policy_optimize_step()
        self._training_step += 1

        # check if it's time to update the reference policy
        if (
            self.ctx.hparams.update_ref_policy_every_n_steps > 0
            and self._training_step % self.ctx.hparams.update_ref_policy_every_n_steps
            == 0
        ):
            self.update_reference_policy()

        return gradnorm

    def train(self):
        # initial eval before we start
        # self.eval_model()

        # now we iterate
        pbar = tqdm(
            total=self.ctx.hparams.max_steps,
            desc="Training",
            initial=self._training_step,
        )

        train_iter = iter(self.train_loader)
        while self._training_step < self.ctx.hparams.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                # Reset the iterator when we've exhausted the train_loader
                train_iter = iter(self.train_loader)

                # evaluate the model at the end of each pass through the dataset
                self.eval_model()
                dist.barrier()

                # save a checkpoint
                self.ctx.save_checkpoint(f"step_{self._training_step}")

                batch = next(train_iter)

            # here we need to create a set of rollouts for each prompt
            rollout_msl = (
                self.ctx.hparams.msl_pre
                if self._training_step < self.ctx.hparams.msl_jump_at_step
                else self.ctx.hparams.msl_post
            )
            rollouts = self.simple_rollouts(batch, max_seq_len=rollout_msl)
            dist.barrier()

            # Print sample inputs/outputs for monitoring
            self._print_sample_rollouts(rollouts, num_samples=2)

            # Calculate average reward, accuracy, and parse rate for this batch
            total_rewards = sum(
                rollout.score.reward
                for sample in rollouts
                for rollout in sample.rollouts
            )
            total_correct = sum(
                1
                for sample in rollouts
                for rollout in sample.rollouts
                if rollout.score.is_correct
            )
            total_parsable = sum(
                1
                for sample in rollouts
                for rollout in sample.rollouts
                if rollout.score.is_parsable
            )
            total_rollouts = sum(len(sample.rollouts) for sample in rollouts)
            avg_reward = total_rewards / total_rollouts if total_rollouts > 0 else 0.0
            avg_accuracy = total_correct / total_rollouts if total_rollouts > 0 else 0.0
            avg_parse_rate = (
                total_parsable / total_rollouts if total_rollouts > 0 else 0.0
            )

            # Update tqdm postfix with average reward and accuracy
            pbar.set_postfix(
                {
                    "avg_reward": f"{avg_reward:.4f}",
                    "accuracy": f"{avg_accuracy:.2%}",
                    "step": self._training_step,
                }
            )

            # Log to wandb (rank 0 only)
            if dist.get_rank() == 0:
                wandb.log(
                    {
                        "train/avg_reward": avg_reward,
                        "train/avg_accuracy": avg_accuracy,
                        "train/avg_parse_rate": avg_parse_rate,
                        "train/total_rollouts": total_rollouts,
                        "train/step": self._training_step,
                    },
                    step=self._training_step,
                )

            # before we can train, we need to compute the logprobs via offline inference
            self.compute_offline_logprobs(rollouts)

            # now that we've generated G rollouts for our B groups of prompts,
            # we convert this into a dataset and update the trainable policy on it
            self.train_policy_on_rollouts(rollouts)
            dist.barrier()

            # Update progress bar after training step
            pbar.update(1)

        pbar.close()

    def _extract_raw_answer(self, response: str) -> str:
        """Extract raw content from <answer> tags, or return [UNPARSABLE]."""
        matches = answer_pattern.findall(response)
        if matches:
            # Return last match, truncated if too long
            raw = matches[-1].strip()
            return raw[:20] + "..." if len(raw) > 20 else raw
        return "[UNPARSABLE]"

    def _print_sample_rollouts(self, samples: list[Sample], num_samples: int = 1):
        """Print rollout preview with summary table and best rollout details (rank 0 only)."""
        if dist.get_rank() != 0:
            return

        if not samples or not samples[0].rollouts:
            return

        # Show only 1 sample
        sample = samples[0]
        rollouts = sample.rollouts

        log.info("=" * 60)
        log.info("=== Sample Rollout Preview ===")
        log.info(f"[GROUND TRUTH] {sample.problem.answer}")
        log.info("")

        # Sort rollouts by reward descending
        sorted_rollouts = sorted(rollouts, key=lambda r: r.score.reward, reverse=True)
        best_rollout = sorted_rollouts[0]

        # Print rollout summary table
        log.info(f"--- Rollout Summary ({len(rollouts)} rollouts) ---")
        log.info("Rank | Reward | Correct | Parsed Answer        | Last 40 chars")
        log.info(
            "-----|--------|---------|----------------------|------------------------------------------"
        )

        max_rows = 10
        for idx, rollout in enumerate(sorted_rollouts[:max_rows]):
            rank = idx + 1
            marker = " *" if rollout is best_rollout else "  "
            reward = f"{rollout.score.reward:.2f}"
            correct = "Yes" if rollout.score.is_correct else "No"
            parsed = self._extract_raw_answer(rollout.response)
            last_40 = (
                rollout.response[-40:].replace("\n", "↵")
                if len(rollout.response) >= 40
                else rollout.response.replace("\n", "↵")
            )
            log.info(
                f"{rank:>3}{marker} | {reward:>6} | {correct:>7} | {parsed:<20} | {last_40}"
            )

        if len(sorted_rollouts) > max_rows:
            log.info(f"... ({len(sorted_rollouts) - max_rows} more rollouts)")

        log.info("")
        log.info("* = Best rollout (shown below)")
        log.info("")

        # Show best rollout details
        log.info("--- Best Rollout Details ---")
        log.info(f"[PROBLEM] {sample.problem.problem}")
        response_preview = (
            best_rollout.response[:500] + "..."
            if len(best_rollout.response) > 500
            else best_rollout.response
        )
        log.info(f"[OUTPUT]\n{response_preview}")
        log.info("=" * 60)

    @torch.no_grad()
    def compute_offline_logprobs(self, samples: list[Sample]):
        self.ctx.model.eval()
        log_rank_0("waiting for other nodes before starting logprobs computation")
        dist.barrier()

        # this is a super stupid way of handling this computation but let's just do 1 at a time
        # as a proof of concept
        if dist.get_rank() == 0:
            iterator = tqdm(samples, desc="Computing logprobs", total=len(samples))
        else:
            iterator = samples

        for sample in iterator:
            for rollout in sample.rollouts:
                # generate the inputs, all of these are dimension (1, T)
                full_seq = torch.tensor(
                    [sample.input_ids + rollout.token_ids[:-1]],
                    dtype=torch.long,
                    device=self.ctx.device,
                )
                position_ids = torch.tensor(
                    [range(full_seq.shape[-1])],
                    dtype=torch.long,
                    device=self.ctx.device,
                )
                logprob_ids = torch.tensor(
                    [rollout.token_ids], dtype=torch.long, device=self.ctx.device
                )

                # ADD: Create flash attention kwargs for single sequence
                seq_len = full_seq.shape[1]
                flash_attn_kwargs = {
                    "cu_seq_lens_q": torch.tensor(
                        [0, seq_len], dtype=torch.int32, device=self.ctx.device
                    ),
                    "cu_seq_lens_k": torch.tensor(
                        [0, seq_len], dtype=torch.int32, device=self.ctx.device
                    ),
                    "max_length_q": seq_len,
                    "max_length_k": seq_len,
                }

                # [1, 2, 3, 4, 5]
                # [2, 3, 4, 5, 0]

                # compute the output
                outputs = self.ctx.model(
                    input_ids=full_seq,
                    position_ids=position_ids,
                    logits_to_keep=len(rollout.token_ids),
                    **flash_attn_kwargs,
                )
                logits = outputs.logits.float()  # (1, T, V)
                assert len(logits.shape) == 3
                assert logits.shape[1] == len(rollout.token_ids)

                if self.ctx.sampling_params.temperature > 0:
                    logits /= self.ctx.sampling_params.temperature

                # (1, T) -> (1, T, 1)
                logprob_ids = logprob_ids.unsqueeze(-1)

                # now we need to pluck out the logits we care about
                old_logits = torch.gather(
                    logits, dim=-1, index=logprob_ids
                )  # (1, T, 1)
                old_logsumexp = logits.logsumexp(dim=-1, keepdim=True)  # (1, T, 1)
                old_logprobs = (
                    old_logits - old_logsumexp
                )  # should be equivalent to log(p(x))

                # dist.breakpoint()
                # old logprobs: (1, T, 1)

                # (1, T, 1) --> (1, T) --> (T,)
                old_logprobs = old_logprobs.squeeze(-1).squeeze(
                    0
                )  # oh, so old_logprobs here is actually (T, T,) somehow
                assert old_logprobs.shape == (len(rollout.logprobs),), (
                    f"{old_logprobs.shape=} != {(len(rollout.logprobs),)=}"
                )
                old_logprobs_offline = old_logprobs.tolist()

                assert len(old_logprobs_offline) == len(rollout.logprobs), (
                    f"{len(old_logprobs_offline)=} != {len(rollout.logprobs)=}"
                )

                # dist.breakpoint()

                # set logprobs
                for i, t in enumerate(rollout.logprobs):
                    t.logprob = old_logprobs_offline[i]

                assert full_seq.shape == position_ids.shape, (
                    f"{full_seq.shape=} != {position_ids.shape=}"
                )

                # wait for all
                dist.barrier()

    def train_policy_on_rollouts(self, samples: list[Sample]):
        self.ctx.model.train()

        any_rewards = sum(r.score.reward for sample in samples for r in sample.rollouts)

        # Reduce rewards across all ranks
        rewards_tensor = torch.tensor(
            [any_rewards], dtype=torch.float32, device=self.ctx.device
        )
        dist.all_reduce(rewards_tensor, op=dist.ReduceOp.SUM)
        # total_rewards = rewards_tensor.item()

        # Create the distributed data loader with lockstep microbatches
        # This generator yields AccumulationWindow objects
        # NOTE: global_batch_size should be total rollouts (batch_size * group_size) to ensure
        # all rollouts are processed in ONE optimizer step. Previously this used just batch_size,
        # causing 16 optimizer steps per rollout batch instead of 1.
        # total_rollouts = self.ctx.hparams.batch_size * self.ctx.hparams.group_size
        data_loader = create_oleg_grpo_dataloader(
            samples,
            self.ctx.tokenizer.pad_token_id,
            self.ctx.hparams.max_tokens_per_gpu,
            self.ctx.hparams.inner_batch_size,
        )

        # now we train
        for epoch in range(self.ctx.hparams.inner_epochs):
            # Iterate through accumulation windows (each window = K microbatches)
            batch_count = 0
            for minibatch in data_loader:
                batch_count += 1
                log_rank_0(f"[DEBUG] Processing minibatch {batch_count}")
                # at the start of the microbatch, every rank announces the length they are packing to
                num_samples_in_local_minibatch = minibatch[
                    "num_samples_in_local_minibatch"
                ]
                batch_size = torch.tensor(
                    [num_samples_in_local_minibatch],
                    device=torch.device("cuda"),
                    dtype=torch.long,
                )
                dist.all_reduce(batch_size, op=dist.ReduceOp.SUM)
                batch_size = batch_size.item()
                assert batch_size == self.ctx.hparams.inner_batch_size

                batch_structure = [
                    "." if not mb["padding"] else "#"
                    for mb in minibatch["microbatches"]
                ]
                for curr_r in range(dist.get_world_size()):
                    if curr_r == dist.get_rank():
                        print(f"rank {curr_r}: [" + " ".join(batch_structure) + "]")

                    dist.barrier()

                # Track KL divergence across microbatches in this accumulation window
                accum_kl_sum = 0.0
                accum_kl_count = 0
                no_padding = 0
                loss_accum = 0

                # Process K microbatches and accumulate gradients
                for microbatch_idx, microbatch in enumerate(minibatch["microbatches"]):
                    # Handle padding batches - we still need to run forward/backward
                    # to keep FSDP collectives in sync, but we zero out the loss contribution

                    batch = microbatch["microbatch"]

                    """
                    minibatch here has columns input_ids, labels, advantage. it is indexed column-first and row-second
                    """

                    # next step, do a minibatch of rollouts

                    """
                    Okay so the next thing we have to do is make this training function work for all groups
                    and all batches. 
                
                    These are the things we need to fix:
                
                    1. Model needs to consume multiple inputs
                    2. Importance ratio calculated with multiple inputs + advantages
                    3. KL divergence has to be adjusted to work with multiple inputs
                    4. Loss has to be adjusted to work with multiple inputs
                    5. We need to add sequence-level averaging
                        (all rollouts are considered independent, each rollout's token loss is averaged by the number
                        of tokens in that rollout)
    
                    """

                    # send everything to GPU as needed
                    input_ids = batch["input_ids"].to(self.ctx.device).unsqueeze(0)
                    logprob_ids = batch["logprob_ids"].to(self.ctx.device).unsqueeze(0)
                    advantages = batch["advantages"].to(self.ctx.device).unsqueeze(0)
                    old_logprobs = batch["logprobs"].to(self.ctx.device).unsqueeze(0)
                    grpo_logit_mask = (
                        batch["grpo_mask"].to(self.ctx.device).unsqueeze(0)
                    )
                    scalars = batch["scalars"].to(self.ctx.device).unsqueeze(0)
                    position_ids = (
                        batch["position_ids"].to(self.ctx.device).unsqueeze(0)
                    )
                    cu_seq_lens_q = batch["cu_seq_lens_q"].to(self.ctx.device)
                    cu_seq_lens_k = batch["cu_seq_lens_k"].to(self.ctx.device)
                    max_length_q = batch["max_length_q"]
                    max_length_k = batch["max_length_k"]

                    # CRITICAL: In no-packing mode, vLLM uses standard attention (no varlen kwargs)
                    # We should match that behavior to get identical logprobs
                    # Only use varlen FlashAttention when actually packing multiple sequences
                    flash_attn_kwargs = {}
                    flash_attn_kwargs = {
                        "cu_seq_lens_q": cu_seq_lens_q,
                        "cu_seq_lens_k": cu_seq_lens_k,
                        "max_length_q": max_length_q,
                        "max_length_k": max_length_k,
                    }

                    # forward
                    new_outputs = self.ctx.model(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        **flash_attn_kwargs,
                    )

                    B, T, V = new_outputs.logits.shape
                    assert B == 1
                    assert input_ids.shape == (1, T)
                    assert position_ids.shape == (1, T)
                    assert grpo_logit_mask.shape == (1, T)
                    assert old_logprobs.shape == (1, T)
                    assert advantages.shape == (1, T)
                    assert logprob_ids.shape == (1, T)

                    # prepare for GRPO loss calculation, pluck out the logits that we're going to work with
                    gather_indices = logprob_ids.clone().unsqueeze(-1)  # (1, B*T, 1)
                    assert gather_indices.shape == (1, T, 1)
                    # (1, B, V)
                    new_logits = new_outputs.logits.float()
                    assert new_logits.shape == (1, T, V)

                    # account for sampling temperature
                    if self.ctx.sampling_params.temperature > 0:
                        new_logits /= self.ctx.sampling_params.temperature

                    # now let's get the reference logits, but we really want to make sure that we don't
                    # backprop on this model
                    # with torch.no_grad():
                    with torch.no_grad():
                        ref_outputs = self.ctx.ref_model(
                            input_ids, position_ids=position_ids, **flash_attn_kwargs
                        )
                        ref_logits = ref_outputs.logits.float()
                        assert ref_logits.shape == (1, T, V)
                        # log_rank_0(
                        #     f"[GRPO] Step 3: Extract ref logits | ref_outputs.logits: {ref_outputs.logits.shape} -> ref_logits: {ref_logits.shape}"
                        # )
                        if self.ctx.sampling_params.temperature > 0:
                            ref_logits /= self.ctx.sampling_params.temperature
                            # log_rank_0(
                            #     f"[GRPO] Step 3a: Apply temperature to ref | temperature: {self.ctx.sampling_params.temperature} | ref_logits: {ref_logits.shape} (in-place)"
                            # )
                        # 2. Calculate the ref logprobs
                        # with torch.no_grad():
                        ref_gathered_logits = ref_logits.gather(
                            dim=-1, index=gather_indices
                        )
                        assert ref_gathered_logits.shape == (1, T, 1)
                        ref_gathered_logits = ref_gathered_logits.squeeze(-1)
                        assert ref_gathered_logits.shape == (1, T)
                        # log_rank_0(
                        #     f"[GRPO] Step 7: Gather ref logits | ref_logits: {ref_logits.shape}, gather_indices: {gather_indices.shape} -> ref_gathered_logits: {ref_gathered_logits.shape}"
                        # )

                        ref_logsumexp = ref_logits.logsumexp(
                            dim=-1, keepdim=True
                        )  # (B, T, 1)
                        assert ref_logsumexp.shape == (1, T, 1)
                        ref_logsumexp = ref_logsumexp.squeeze(-1)
                        assert ref_logsumexp.shape == (1, T)
                        # log_rank_0(
                        #     f"[GRPO] Step 8: Compute ref logsumexp | ref_logits: {ref_logits.shape} -> ref_logsumexp: {ref_logsumexp.shape}"
                        # )

                        ref_logprobs = ref_gathered_logits - ref_logsumexp
                        assert ref_logprobs.shape == (1, T)
                        # log_rank_0(
                        #     f"[GRPO] Step 9: Compute ref logprobs | ref_gathered_logits: {ref_gathered_logits.shape}, ref_logsumexp: {ref_logsumexp.shape} -> ref_logprobs: {ref_logprobs.shape}"
                        # )

                    # hopefully clears the reference model cache
                    torch.cuda.empty_cache()
                    # 3. Calculate the latest logprobs
                    # more efficient technique
                    # (1, B, V) --> (1, B, 1)
                    new_gathered_logits = new_logits.gather(
                        dim=-1, index=gather_indices
                    )
                    assert new_gathered_logits.shape == (1, T, 1)

                    new_gathered_logits = new_gathered_logits.squeeze(
                        -1
                    )  # (1, B, 1) --> (1, B)
                    assert new_gathered_logits.shape == (1, T)
                    # log_rank_0(
                    #     f"[GRPO] Step 4: Gather new logits | new_logits: {new_logits.shape}, gather_indices: {gather_indices.shape} -> new_gathered_logits: {new_gathered_logits.shape}"
                    # )

                    new_logits: torch.Tensor

                    # (1, B, V) --> (1, B, 1) , computes log(sum(exp(k) for k in logits))
                    new_logsumexp = new_logits.logsumexp(
                        dim=-1, keepdim=True
                    )  # (1, B, 1)
                    assert new_logsumexp.shape == (1, T, 1)
                    new_logsumexp = new_logsumexp.squeeze(-1)  # (1, B, 1) -> (1, B)
                    assert new_logsumexp.shape == (1, T)

                    # so this definition should be correct
                    new_logprobs = new_gathered_logits - new_logsumexp  # (1, B)
                    assert new_logprobs.shape == (1, T)
                    assert len(new_logprobs.shape) == 2

                    # 4. Compute the importance ratio
                    # \rho(\theta_0)=\exp(\log p_{\theta_0}-\log p_{\text{old}})
                    assert old_logprobs.shape == new_logprobs.shape

                    B, T = new_logprobs.shape
                    assert B == 1

                    importance_ratio: torch.Tensor = (
                        (new_logprobs - old_logprobs).clamp(-20, 20).exp()
                    )
                    assert importance_ratio.shape == (1, T)

                    # (1,B) * (1,B)
                    unclipped = advantages * importance_ratio
                    assert importance_ratio.shape == (1, T), (
                        f"{importance_ratio.shape} != {(1, T)}"
                    )
                    assert advantages.shape == (1, T), f"{advantages.shape} != {(1, T)}"
                    assert unclipped.shape == (1, T), f"{unclipped.shape} != {(1, T)}"

                    # (1,B) * (1,B)
                    clipped_ratio = importance_ratio.clamp(
                        1 - self.ctx.hparams.eps, 1 + self.ctx.hparams.eps
                    )
                    assert clipped_ratio.shape == (1, T)
                    clipped = advantages * clipped_ratio
                    assert clipped_ratio.shape == (1, T)

                    clipped_surrogate = torch.minimum(unclipped, clipped)
                    assert clipped_surrogate.shape == (1, T)

                    logprob_diff = (ref_logprobs - new_logprobs).clamp(-20, 20)
                    assert logprob_diff.shape == (1, T)

                    dkl_approx = logprob_diff.exp() - logprob_diff - 1
                    dkl_approx = dkl_approx.clamp(0, 100)
                    assert dkl_approx.shape == (1, T)

                    # Accumulate KL for logging (only for non-padding batches)
                    if not microbatch["padding"]:
                        no_padding += 1

                    masked_kl = dkl_approx * grpo_logit_mask.float()
                    assert masked_kl.shape == (1, T)
                    assert scalars.shape == (1, T)
                    avg_kl = (masked_kl / scalars).sum() / batch_size

                    accum_kl_sum += avg_kl
                    assert dkl_approx.shape == clipped_surrogate.shape, (
                        f"{dkl_approx.shape=} != {clipped_surrogate.shape=}"
                    )
                    per_token_loss = (
                        clipped_surrogate
                        - self.ctx.hparams.kl_penalty_strength * dkl_approx
                    )
                    assert per_token_loss.shape == (1, T)
                    # 8. Mask out all invalid logprobs that aren't from the GRPO rollouts
                    assert grpo_logit_mask.shape == (1, T)
                    grpo_token_loss = per_token_loss * grpo_logit_mask.float()
                    assert grpo_token_loss.shape == (1, T)
                    # this achieves length averaging.
                    if not self.ctx.hparams.dr_grpo:
                        assert grpo_token_loss.shape == scalars.shape
                        grpo_token_loss = grpo_token_loss / scalars.float()

                    # this should achieve the group average
                    # we need to multiply our loss by the world size because FSDP2 will try to average the loss by the world size
                    # so we just want the token loss across all ranks [(a1 + ... + ak) + (b1+ ...+)] / batch size
                    # assert dist.get_world_size() == 1  # debug
                    grpo_sequence_loss = (
                        grpo_token_loss.sum(dim=-1) * dist.get_world_size()
                    ) / batch_size
                    assert grpo_sequence_loss.shape == (1,)

                    grpo_loss = -grpo_sequence_loss
                    assert grpo_loss.shape == (1,)
                    # For padding batches, zero out the loss so it doesn't contribute to gradients
                    # We still run forward/backward to keep FSDP collectives in sync
                    if microbatch["padding"]:
                        grpo_loss = grpo_loss * 0.0
                        # log_rank_0(f"[GRPO] Step 22: Zero padding loss | grpo_loss: {grpo_loss.shape} (in-place)")
                    #         log_rank_0("=" * 120 + "\n")

                    log_rank_0(f"{grpo_token_loss.sum()=}, {grpo_loss=}")

                    # backprop (accumulate gradients)
                    # dist.breakpoint()
                    grpo_loss.backward()
                    log_rank_0(f"{grpo_token_loss.sum()=}, {grpo_loss=}")

                    # log_rank_0(f"[GRPO] Step 23: Backward pass | grpo_loss: {grpo_loss.shape}")

                    # after backward(), grpo_loss now contains the total loss averaged across all ranks
                    with torch.no_grad():
                        loss_accum += grpo_loss.item()

                    # Log per-microbatch stats
                    torch.cuda.empty_cache()

                    # dist.breakpoint()

                # After processing all K microbatches in this window, take optimizer step
                gradnorm = self.take_training_step()

                # Calculate average KL for this trianing step
                # TODO: this isn't fully correct, because padded training steps will need to contribute 0 KL,
                # but this also isn't the reason that the algorithm is struggling to learn right now.
                average_kl = accum_kl_sum
                log_rank_0(
                    f"Optimizer Step {self._training_step} | Grad Norm: {gradnorm.item():.4f} | Avg KL: {avg_kl:.6f}"
                )

                # Log grad norm and KL to wandb
                if dist.get_rank() == 0:
                    wandb.log(
                        {
                            "train/loss": loss_accum.item()
                            if isinstance(loss_accum, torch.Tensor)
                            else loss_accum,
                            "train/grad_norm": gradnorm.item(),
                            "train/kl_divergence": average_kl,
                        },
                        step=self._training_step,
                    )
                dist.barrier()
                # dist.breakpoint()

                # DIAGNOSTIC: Breakpoint to manually advance through training steps
                # dist.breakpoint()


@app.command()
def train(
    # we need to know where to load the model from
    model_dir: str = typer.Option(..., help="Path where the model will be loaded from"),
    # for inference
    vllm_url: str = typer.Option(
        "http://localhost:8000/v1", "--vllm-url", help="vLLM endpoint"
    ),
    vllm_model_name: str = typer.Option(
        ..., "--vllm-model-name", help="Name of the model being hosted in vLLM"
    ),
    vllm_model_dir: str = typer.Option(
        ...,
        "--vllm-model-dir",
        help="Directory where vLLM expects to reload the model from",
    ),
    # dataset parameters, we'll eventually move these to a data generation command
    train_path: str | None = typer.Option(
        None, "--train-path", help="Path to the training data"
    ),
    dataset: str = typer.Option(
        "gsm8k",
        "--dataset",
        help="the dataset to use for training. Use 'json' for reading from local files in the 'problem'/'answer' format",
    ),
    seed: int = typer.Option(67, help="Random seed"),
    num_problems: int = typer.Option(20, help="Number of problems"),
    min_num: int = typer.Option(-100, help="Minimum number for problems"),
    max_num: int = typer.Option(100, help="Maximum number for problems"),
    system_msg: str | None = typer.Option(
        None, "--system-msg", help="System message to prompt the model"
    ),
    # model
    # training params
    max_new_tokens: int = typer.Option(
        128, help="The maximum number of new tokens that the model can generate."
    ),
    msl_post: int = typer.Option(
        2**16,
        help="maximum length of the sequences that we work with",
    ),
    msl_pre: int = typer.Option(
        2**15,
        help="maximum length of the sequences that we work with",
    ),
    max_tokens_per_gpu: int = typer.Option(
        2**11,
        help="This is the maximum amount of tokens we can use at a time during training.",
    ),
    max_steps: int = typer.Option(10400, help="Number of training epochs"),
    msl_jump_at=typer.Option(8200, help="Number of training epochs"),
    # adamw parms
    lr: float = typer.Option(1e-5, "--lr", help="Learning rate"),
    beta1: float = typer.Option(0.9, help="Adam beta1 parameter"),
    beta2: float = typer.Option(0.95, help="Adam beta2 parameter"),
    wd: float = typer.Option(0.0, "--wd", help="Weight decay"),
    # device selection
    # GRPO params
    inner_epochs: int = typer.Option(1, help="Number of passes on inner generation"),
    inner_batch_size: int = typer.Option(
        4, "--inner-batch-size", help="Batch size during the GRPO inner loop."
    ),
    batch_size: int = typer.Option(
        1,
        "-B",
        "--batch-size",
        help="Number of prompts to batch together when generating GRPO rollouts.",
    ),
    group_size: int = typer.Option(
        1,
        "-G",
        "--group-size",
        help="Group size / number of rollouts to generate from a single prompt",
    ),
    ref_update_frequency: int = typer.Option(
        -1,
        "--ref-update-frequency",
        help="The frequency (in steps) in which the reference policy is updated.",
    ),
    ref_cpu_offload: bool = typer.Option(
        False,
        "--ref-cpu-offload",
        help="CPU offload for reference model to save GPU memory",
        is_flag=True,
    ),
    temperature: float = typer.Option(0.7, "-t", "--temp", help="sampling temperature"),
    clip_eps: float = typer.Option(
        0.1, "--clip-eps", help="epsilon used for GRPO clip"
    ),
    kl_strength: float = typer.Option(
        0.01, "--kl", help="strength of the kl penalty to the reference policy"
    ),
    dr_grpo: bool = typer.Option(
        False, "--dr-grpo", help="Whether to use the DR-GRPO algorithm", is_flag=True
    ),
    # eval split
    eval_split: float = typer.Option(
        0.0,
        "--eval-split",
        help="portion of training samples to use for the eval dataset",
    ),
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        help="Optional directory to use for saving the checkpoints",
    ),
    reward_fn: str = typer.Option(
        "math_with_thinking",
        "--reward-fn",
        help="Reward function to use. Options: math_with_thinking, accuracy_only, format_only, strict_format_with_accuracy",
    ),
):
    # Set the reward function before anything else
    import src.rewards as rewards_module

    rewards_module.DEFAULT_REWARD_FN = rewards_module.get_reward_fn(reward_fn)
    log.info(f"Using reward function: {reward_fn}")

    # initialize distributed
    init_distributed()

    # Initialize wandb on rank 0 only
    if dist.get_rank() == 0:
        wandb.init(
            project="mini-grpo",
            config={
                "model": model_dir,
                "dataset": dataset,
                "lr": lr,
                "max_steps": max_steps,
                "batch_size": batch_size,
                "group_size": group_size,
                "inner_epochs": inner_epochs,
                "inner_batch_size": inner_batch_size,
                "clip_eps": clip_eps,
                "kl_strength": kl_strength,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "msl_pre": msl_pre,
                "msl_post": msl_post,
                "msl_jump_at": msl_jump_at,
                "ref_update_frequency": ref_update_frequency,
                "max_tokens_per_gpu": max_tokens_per_gpu,
                "reward_fn": reward_fn,
                "world_size": dist.get_world_size(),
            },
        )

    # device setup
    # create training components
    train_ctx = TrainingContext(
        model_name=model_dir,
        system_msg=system_msg,
        vllm_url=vllm_url,
        vllm_model_name=vllm_model_name,
        vllm_model_dir=vllm_model_dir,
        hparams=Hyperparameters(
            lr=lr,
            max_steps=max_steps,
            msl_post=msl_post,
            msl_pre=msl_pre,
            msl_jump_at_step=msl_jump_at,
            batch_size=batch_size,
            group_size=group_size,
            inner_epochs=inner_epochs,
            inner_batch_size=inner_batch_size,
            dr_grpo=dr_grpo,
            eps=clip_eps,
            kl_penalty_strength=kl_strength,
            adamw_betas=(beta1, beta2),
            adamw_wd=wd,
            update_ref_policy_every_n_steps=ref_update_frequency,
            max_tokens_per_gpu=max_tokens_per_gpu,
        ),
        sampling_params=SamplingParams(
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=1.0,
            top_k=0.0,
            repetition_penalty=1.0,
        ),
        output_dir=output_dir,
        train_path=train_path,
        eval_split=eval_split,
        world_size=dist.get_world_size(),
        dataset=dataset,
        ref_model_cpu_offload=ref_cpu_offload,
    )
    trainer = GRPOTrainer(train_ctx)

    dist.barrier()
    trainer.initialize()
    trainer.train()

    # Finish wandb run on rank 0
    if dist.get_rank() == 0:
        wandb.finish()

    destroy_distributed_environment()


def stream_output(stream, prefix, file=sys.stdout):
    for line in iter(stream.readline, b""):
        print(f"[{prefix}] {line.decode().rstrip()}", file=file, flush=True)


@app.command()
def orchestrator(
    model_write_dir: str = "/dev/shm/mini-r1/current-model",
    checkpoint_dir: str | None = typer.Option(
        None, "--checkpoint-dir", help="Directory to save the checkpoint in"
    ),
    train_gpus: int = typer.Option(
        6, "--train-gpus", help="Num GPUs to be used for training"
    ),
    vllm_gpus: int = typer.Option(
        2, "--vllm-gpus", help="Num GPUs to be used for the vLLM server"
    ),
    use_olmo: bool = typer.Option(
        False,
        "--use-olmo",
        help="Whether or not to use the Olmo 7B pretrained base (for replicating R1 zero)",
        is_flag=True,
    ),
    model_path: str | None = typer.Option(
        None,
        "--model-path",
        help="Path or name of the model to train with if not using olmo",
    ),
    system_msg: str | None = typer.Option(
        "You're a helpful assistant who helps the user solve challenging problems. Always provide your final answer within <answer>...</answer> tags.",
        "--system-msg",
        help="System message to prompt the model",
    ),
    kl_strength: float = typer.Option(0.01, "--kl", help="Strength of the KL penalty"),
    # batch_size: int = typer.Option(288, "--batch-size", help="Batch size (for generation)"),
    # batch_size: int = typer.Option(54, "--batch-size", help="Batch size (for generation)"),
    # defaults are based on R1 Zero pipeline. R1 Zero used a batch size of 512, we have 6 GPUs so use 510
    batch_size: int = typer.Option(
        48, "--batch-size", help="Batch size (for generation)"
    ),
    group_size: int = typer.Option(
        16, "--group-size", help="Rollout size (num rollouts to generate/batch)"
    ),
    inner_batch_size: int = typer.Option(
        32, "--inner-batch-size", help="Inner batch size for the GRPO update step"
    ),
    inner_epochs: int = typer.Option(
        2, "--inner-epochs", help="Number of inner epochs"
    ),
    clip_eps: float = typer.Option(0.1, "--clip-eps", help="Epsilon for the GRPO clip"),
    epochs: int = typer.Option(10, "--epochs", help="Number of epochs"),
    ref_update_frequency: int = typer.Option(
        400,
        "--ref-update-frequency",
        help="The frequency (in steps) in which the reference policy is updated.",
    ),
    max_tokens_per_gpu: int = typer.Option(
        2**11, "--max-tokens-per-gpu", help="Maximum tokens per GPU"
    ),
    msl_post: int = typer.Option(
        2**12,
        help="maximum length of the sequences that we work with",
    ),
    msl_pre: int = typer.Option(
        2**11,
        help="maximum length of the sequences that we work with",
    ),
    max_steps: int = typer.Option(10400, help="Number of training epochs"),
    max_new_tokens: int = typer.Option(
        128, help="The maximum number of new tokens that the model can generate."
    ),
    msl_jump_at=typer.Option(8200, help="Number of training epochs"),
    verbose_vllm: bool = typer.Option(False),
    reward_fn: str = typer.Option(
        "math_with_thinking",
        "--reward-fn",
        help="Reward function to use. Options: math_with_thinking, accuracy_only, format_only, strict_format_with_accuracy, gsm8k",
    ),
    lr: float = typer.Option(1e-5, "--lr", help="Learning rate to use during training"),
    ref_cpu_offload: bool = typer.Option(
        False,
        "--ref-cpu-offload",
        help="CPU offload for reference model to save GPU memory",
        is_flag=True,
    ),
    dr_grpo: bool = typer.Option(
        False, "--dr-grpo", help="Whether to use the DR-GRPO algorithm", is_flag=True
    ),
):
    # # first load and write the model
    if use_olmo:
        prepare_model(model_write_dir, model_path, use_olmo)
    elif model_path is None:
        raise ValueError("--model-path must be specified if not using olmo")
    else:
        prepare_model(model_write_dir, model_path)

    served_name = "current-policy"

    vllm_env = os.environ.copy()
    train_env = os.environ.copy()
    vllm_env.update(
        {
            "VLLM_SERVER_DEV_MODE": "1",
            "CUDA_VISIBLE_DEVICES": ",".join(str(k) for k in range(vllm_gpus + 1)),
        }
    )
    train_gpu_ids = ",".join([str(k) for k in range(vllm_gpus, vllm_gpus + train_gpus)])
    train_env.update(
        {
            "CUDA_VISIBLE_DEVICES": train_gpu_ids,
        }
    )

    vllm_process, training_process = None, None
    try:
        vllm_cmd = [
            "vllm",
            "serve",
            model_write_dir,
            "--enable-sleep-mode",
            "--served-model-name",
            served_name,
            "--logprobs-mode",
            "processed_logprobs",  # returned logprobs will have temp. scaling
            "--port",
            "8000",
            # use 2 gpus for inference for right now
            "--data-parallel-size",
            str(vllm_gpus),
            "--dtype",
            "float16",  # Match training dtype for consistent logprobs
        ]
        if not verbose_vllm:
            vllm_cmd += ["--disable-log-requests", "--uvicorn-log-level", "warning"]
        vllm_process = subprocess.Popen(
            # [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model_write_dir, "--enable-sleep-mode"],
            vllm_cmd,
            env=vllm_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # separate pipe for stderr
            bufsize=1,
        )

        # launch training with captured output
        train_cmd = [
            "torchrun",
            "--nproc-per-node",
            str(train_gpus),
            "cli.py",
            "train",
            "--train-path",
            "generated_data/train.jsonl",
            "--model-dir",
            model_write_dir,
            "--vllm-model-dir",
            model_write_dir,
            "--vllm-model-name",
            served_name,
            "--vllm-url",
            "http://localhost:8000",
            "--eval-split",
            "0.25",
            "--batch-size",
            str(batch_size),
            "--group-size",
            str(group_size),
            "--inner-batch-size",
            str(inner_batch_size),
            "--inner-epochs",
            str(epochs),
            "--ref-update-frequency",
            str(ref_update_frequency),
            "--msl-pre",
            str(msl_pre),
            "--msl-post",
            str(msl_post),
            "--max-steps",
            str(max_steps),
            "--msl-jump-at",
            str(msl_jump_at),
            "--max-new-tokens",
            str(max_new_tokens),
            "--max-tokens-per-gpu",
            str(max_tokens_per_gpu),
            "--reward-fn",
            reward_fn,
            "--lr",
            str(lr),
            "--kl",
            str(kl_strength),
            "--inner-epochs",
            str(inner_epochs),
            "--clip-eps",
            str(clip_eps),
        ]
        if checkpoint_dir:
            train_cmd += ["--output-dir", checkpoint_dir]
        if system_msg:
            train_cmd += ["--system-msg", system_msg]
        if ref_cpu_offload:
            train_cmd += ["--ref-cpu-offload"]
        if dr_grpo:
            train_cmd += ["--dr-grpo"]
        training_process = subprocess.Popen(
            train_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=train_env,
            bufsize=1,
        )

        # stream stdout
        vllm_thread = threading.Thread(
            target=stream_output, args=(vllm_process.stdout, "VLLM"), daemon=True
        )
        training_thread = threading.Thread(
            target=stream_output, args=(training_process.stdout, "TRAIN"), daemon=True
        )

        vllm_thread.start()
        training_thread.start()

        # wait for training to complete
        training_process.wait()

    except Exception as e:
        log_rank_0(f"encountered error while launching training jobs: {e}")
    finally:
        # Clean up processes if they exist
        if vllm_process and vllm_process.poll() is None:
            log_rank_0("Terminating vLLM process...")
            vllm_process.terminate()
            try:
                vllm_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                log_rank_0("Force killing vLLM process...")
                vllm_process.kill()

        if training_process and training_process.poll() is None:
            log_rank_0("Terminating training process...")
            training_process.terminate()
            try:
                training_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                log_rank_0("Force killing training process...")
                training_process.kill()


if __name__ == "__main__":
    app()
