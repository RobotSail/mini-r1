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

from data_utils import (
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
)
from utils import display_scorecard, init_distributed, destroy_distributed_environment, log_rank_0
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
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions, set_model_state_dict

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
    eval_path: str = typer.Option(..., "--eval-path", help="Path to the evaluation dataset (jsonl)"),
    model_name: str = typer.Option(..., "--model", "-m", help="Model name or path"),
    gpu: int = typer.Option(0, "--gpu", "-g", help="CUDA GPU index to use"),
    max_new_tokens: int = typer.Option(128, help="Maximum number of new tokens to generate"),
    max_seq_len: int = typer.Option(8192, "--msl", "--max-seq-len", help="Maximum sequence length"),
    temperature: float = typer.Option(0.7, "-t", "--temp", help="Sampling temperature"),
    group_size: int = typer.Option(1, "-G", "--group-size", help="Number of rollouts per prompt (for pass@k)"),
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

    # samples = generate_rollouts(
    #     model,
    #     tokenizer,
    #     batch=next(iter(eval_data)),
    #     batch_size=eval_dataset.num_rows,
    #     group_size=group_size,
    #     sampling_params=sampling_params,
    #     show_tqdm=True,
    # )

    percent_scores = []
    for sample in samples:
        passing_rate = sum(1 if r.is_correct else 0 for r in sample.rollouts) / len(sample.rollouts)
        percent_scores.append(passing_rate)

    percent_above_50 = sum(1 if score > 0.5 else 0 for score in percent_scores) / len(percent_scores) * 100
    percent_at_100 = sum(1 if score == 1.0 else 0 for score in percent_scores) / len(percent_scores) * 100

    log.info("=== Evaluation Results ===")
    log.info(f"Model: {model_name}")
    log.info(f"Samples: {len(samples)}")
    log.info(f"Pass@{group_size}: {percent_above_50:.1f}% above 50% | {percent_at_100:.1f}% at 100%")


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
            self.train_dataset, self.eval_dataset = ProblemDataset.from_jsonl(ctx.train_path, ctx.eval_split)
        elif ctx.dataset == "gsm8k":
            self.train_dataset, self.eval_dataset = ProblemDataset.from_gsm8k(ctx.eval_split)
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
        self.ctx.wrap_models_with_fsdp2()

        # Create the train loader now that distributed is initialized
        self.train_loader = self._create_problem_data_loader(self.train_dataset, self.ctx.hparams.batch_size)

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
        return DataLoader(dataset, sampler=train_sampler, batch_size=batchsize, collate_fn=problem_collate_fn)

    @torch.no_grad
    def simple_rollouts(self, batch: list[Problem], max_seq_len: int, is_eval=False, n_completions=None):
        log_rank_0("making call to update vLLM policy")
        self.ctx.update_vllm_policy()
        if not is_eval:
            return self.ctx.generate_completions(batch, max_seq_len=max_seq_len)

        kwargs = {}
        if n_completions is not None and n_completions > 0:
            kwargs["n_completions"] = n_completions
        return self.ctx.generate_completions(
            batch, max_seq_len=max_seq_len, include_logprobs=False, only_run_on_main=True, **kwargs
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
            samples = self.simple_rollouts(eval_problems, max_seq_len=rollout_msl, is_eval=True, n_completions=npass)

            # now we go and determine the passing rate
            percent_scores = []
            for sample in samples:
                passing_rate = sum(1 if r.score.is_correct else 0 for r in sample.rollouts) / len(sample.rollouts)
                percent_scores.append(passing_rate)
            # Calculate statistics
            percent_above_50 = (
                sum(1 if score > 0.5 else 0 for score in percent_scores) / max(len(percent_scores), 1) * 100
            )
            percent_at_100 = (
                sum(1 if score == 1.0 else 0 for score in percent_scores) / max(len(percent_scores), 1) * 100
            )

            results.append((npass, percent_above_50, percent_at_100))

        # Print all results at the end
        log.info("=== Evaluation Scorecard ===")
        log.info(f"Total samples evaluated: {len(samples)}")
        for npass, percent_above_50, percent_at_100 in results:
            log.info(
                f"Pass@{npass}: {percent_above_50:.1f}% above 50% | {percent_at_100:.1f}% at 100% (across {len(samples)} samples with {npass} rollout(s) each)"
            )

    def create_grpo_data_loader(
        self,
        samples: list[Sample],
    ):
        """
        Create a distributed data loader for GRPO training with lockstep microbatches.

        This uses FFD packing with K-synchronization to ensure all ranks have
        exactly the same number of microbatches, preventing FSDP collective mismatches.
        """
        from functools import partial

        # Create the collate function with pad token
        _collate_fn = partial(grpo_collate_fn, pad_token_id=self.ctx.tokenizer.pad_token_id)

        # Create the JsonlDataset from samples
        ds = JsonlDataset(dataset=samples, pad_token_id=self.ctx.tokenizer.pad_token_id)

        # Use the distributed packing dataloader
        # This ensures all ranks do exactly K microbatches
        return create_distributed_grpo_dataloader(
            dataset=ds,
            max_tokens_per_rank=self.ctx.hparams.max_tokens_per_gpu,
            device=self.ctx.device,
            collate_fn=_collate_fn,
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
            self.ctx.ref_model, policy_sd, options=StateDictOptions(ignore_frozen_params=False, full_state_dict=False)
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
            and self._training_step % self.ctx.hparams.update_ref_policy_every_n_steps == 0
        ):
            self.update_reference_policy()

        return gradnorm

    def train(self):
        # initial eval before we start
        self.eval_model()

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

            # Calculate average reward for this batch
            total_rewards = sum(rollout.score.reward for sample in rollouts for rollout in sample.rollouts)
            total_rollouts = sum(len(sample.rollouts) for sample in rollouts)
            avg_reward = total_rewards / total_rollouts if total_rollouts > 0 else 0.0

            # Update tqdm postfix with average reward
            pbar.set_postfix({"avg_reward": f"{avg_reward:.4f}", "step": self._training_step})

            # now that we've generated G rollouts for our B groups of prompts,
            # we convert this into a dataset and update the trainable policy on it
            self.train_policy_on_rollouts(rollouts)
            dist.barrier()

            # Update progress bar after training step
            pbar.update(1)

        pbar.close()

    def train_policy_on_rollouts(self, samples: list[Sample]):
        self.ctx.model.train()

        # Create the distributed data loader with lockstep microbatches
        # This generator ensures all ranks have exactly K microbatches
        data_loader = list(self.create_grpo_data_loader(samples))

        # now we train
        for epoch in range(self.ctx.hparams.inner_epochs):
            # Iterate through the microbatches
            for packed_batch in data_loader:
                packed_batch: PackedBatch

                # Handle padding batches - we still need to run forward/backward
                # to keep FSDP collectives in sync, but we zero out the loss contribution
                if packed_batch.is_padding:
                    log_rank_0("padding microbatch - running dummy forward/backward for FSDP sync")

                batch = packed_batch.batch
                if batch is None:
                    # Edge case: completely empty dataset
                    dist.barrier()
                    continue

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
                grpo_logit_mask = batch["grpo_mask"].to(self.ctx.device).unsqueeze(0)
                scalars = batch["scalars"].to(self.ctx.device).unsqueeze(0)
                position_ids = batch["position_ids"].to(self.ctx.device).unsqueeze(0)
                batch_size = batch["batch_size"]

                # we're not calculating cross-entropy against static labels, so no label inputs
                # are needed here
                # print("just before calling model.forward")
                # dist.breakpoint()
                new_outputs = self.ctx.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                )
                # print("just after calling model.forward")
                # dist.breakpoint()

                # (1, B, V)
                new_logits = new_outputs.logits

                # account for sampling temperature
                if self.ctx.sampling_params.temperature > 0:
                    new_logits /= self.ctx.sampling_params.temperature

                # now let's get the reference logits, but we really want to make sure that we don't
                # backprop on this model
                with torch.no_grad():
                    ref_outputs = self.ctx.ref_model(input_ids, position_ids=position_ids)
                    ref_logits = ref_outputs.logits
                    if self.ctx.sampling_params.temperature > 0:
                        ref_logits /= self.ctx.sampling_params.temperature

                # we need the new logprobs
                # new logprobs will be of size (1, B, V), so IDS need to be of shape (1, B, 1)
                # logprob IDS are (1, B), so we unsqueeze into V

                # prepare for GRPO loss calculation, pluck out the logits that we're going to work with
                # 1. create the indices
                gather_indices = logprob_ids.clone().unsqueeze(-1)  # (1, B*T, 1)

                # 2. Calculate the latest logprobs
                # more efficient technique
                new_gathered_logits = new_logits.gather(dim=-1, index=gather_indices)
                new_logsumexp = new_logits.logsumexp(dim=-1, keepdim=True)  # (1, B, 1)
                new_logprobs = (new_gathered_logits - new_logsumexp).squeeze(-1)  # (1, B)
                assert len(new_logprobs.shape) == 2

                # 3. Calculate the ref logprobs
                with torch.no_grad():
                    ref_gathered_logits = ref_logits.gather(dim=-1, index=gather_indices)
                    ref_logsumexp = ref_logits.logsumexp(dim=-1, keepdim=True)  # (B, T, 1)
                    ref_logprobs = (ref_gathered_logits - ref_logsumexp).squeeze(-1)

                # 4. Compute  te importance ratio
                # here's where we need to actually compute the loss. first thing we need is to calculate
                # the importance ratio:
                # \rho(\theta_0)=\exp(\log p_{\theta_0}-\log p_{\text{old}})
                assert old_logprobs.shape == new_logprobs.shape
                importance_ratio: torch.Tensor = (new_logprobs - old_logprobs).exp()
                # log_rank_0(f"{importance_ratio.shape=}")

                # 5. Next we calculate the clipped surrogate objective
                # adjust advantage so it can broadcast cleanly
                # advantages = advantages.unsqueeze(-1)  # (B,) --> (B, 1)
                # advantages is already (1, B) so no need to unsqueeze
                # log_rank_0(f"{advantages.shape=}")

                # (B, 1) * (B, T)

                # (1,B) * (1,B)
                unclipped = advantages * importance_ratio
                # (1,B) * (1,B)
                clipped = advantages * importance_ratio.clamp(1 - self.ctx.hparams.eps, 1 + self.ctx.hparams.eps)
                # log_rank_0(f"{advantages.shape=}")
                # log_rank_0(f"{clipped.shape=}")
                # log_rank_0(f"{unclipped.shape=}")
                clipped_surrogate = torch.minimum(unclipped, clipped)
                # log_rank_0(f"{clipped_surrogate.shape=}")
                # dist.breakpoint()

                # 6. Next, we need to calculate the approximate KL penalty to the ref policy
                # KL(policy_new || policy_ref) ~= policy_ref/policy_new - log(policyref/policy_new) - 1
                dkl_approx = (ref_logprobs - new_logprobs).exp() - (ref_logprobs - new_logprobs) - 1
                # log_rank_0(f"{ref_logprobs.shape=}")
                # log_rank_0(f"{new_logprobs.shape=}")
                # log_rank_0(f"{dkl_approx.shape=}")

                # 7. Compute per-token loss L_GRPO = L_clip - L_kl
                # compute the per-sample loss (loss for all samples is independent at this point)
                assert dkl_approx.shape == clipped_surrogate.shape, f"{dkl_approx.shape=} != {clipped_surrogate.shape=}"
                per_token_loss = clipped_surrogate - self.ctx.hparams.kl_penalty_strength * dkl_approx

                # 8. Mask out all invalid logprobs that aren't from the GRPO rollouts
                grpo_token_loss = per_token_loss * grpo_logit_mask.float()

                # --- loss aggregation ---
                # 9. Next, we average each batch by the **sequence length**, then we average by the group-size
                # Warning: sequence-length averaging assigns greater weight to shorter sequences than longer ones
                # but in our case this is fine, since we only care about an objective reward
                # (B,) --> (B, 1)

                # now we divide each token loss by the number of logprobs
                # this achieves length averaging.
                grpo_token_loss = grpo_token_loss / scalars.float()

                # this should achieve the group average
                grpo_sequence_loss = grpo_token_loss.sum(dim=-1) / batch_size
                grpo_loss = -grpo_sequence_loss

                # For padding batches, zero out the loss so it doesn't contribute to gradients
                # We still run forward/backward to keep FSDP collectives in sync
                if packed_batch.is_padding:
                    grpo_loss = grpo_loss * 0.0

                # backprop
                grpo_loss.backward()

                # we optimize the model
                gradnorm = self.take_training_step()
                if not packed_batch.is_padding:
                    log_rank_0(
                        f"Step {self._training_step} | "
                        f"Inner Epoch {epoch + 1}/{self.ctx.hparams.inner_epochs} | "
                        f"Loss: {grpo_loss.item() / dist.get_world_size():.4f} | "
                        f"Grad Norm: {gradnorm.item():.4f}"
                    )
                dist.barrier()


@app.command()
def train(
    # we need to know where to load the model from
    model_dir: str = typer.Option(..., help="Path where the model will be loaded from"),
    # for inference
    vllm_url: str = typer.Option("http://localhost:8000/v1", "--vllm-url", help="vLLM endpoint"),
    vllm_model_name: str = typer.Option(..., "--vllm-model-name", help="Name of the model being hosted in vLLM"),
    vllm_model_dir: str = typer.Option(
        ..., "--vllm-model-dir", help="Directory where vLLM expects to reload the model from"
    ),
    # dataset parameters, we'll eventually move these to a data generation command
    train_path: str | None = typer.Option(None, "--train-path", help="Path to the training data"),
    dataset: str = typer.Option(
        "gsm8k",
        "--dataset",
        help="the dataset to use for training. Use 'json' for reading from local files in the 'problem'/'answer' format",
    ),
    seed: int = typer.Option(67, help="Random seed"),
    num_problems: int = typer.Option(20, help="Number of problems"),
    min_num: int = typer.Option(-100, help="Minimum number for problems"),
    max_num: int = typer.Option(100, help="Maximum number for problems"),
    # model
    # training params
    max_new_tokens: int = typer.Option(128, help="The maximum number of new tokens that the model can generate."),
    msl_post: int = typer.Option(
        2**16,
        help="maximum length of the sequences that we work with",
    ),
    msl_pre: int = typer.Option(
        2**15,
        help="maximum length of the sequences that we work with",
    ),
    max_tokens_per_gpu: int = typer.Option(
        2**11, help="This is the maximum amount of tokens we can use at a time during training."
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
    inner_batch_size: int = typer.Option(4, "--inner-batch-size", help="Batch size during the GRPO inner loop."),
    batch_size: int = typer.Option(
        1, "-B", "--batch-size", help="Number of prompts to batch together when generating GRPO rollouts."
    ),
    group_size: int = typer.Option(
        1, "-G", "--group-size", help="Group size / number of rollouts to generate from a single prompt"
    ),
    ref_update_frequency: int = typer.Option(
        -1, "--ref-update-frequency", help="The frequency (in steps) in which the reference policy is updated."
    ),
    temperature: float = typer.Option(0.7, "-t", "--temp", help="sampling temperature"),
    clip_eps: float = typer.Option(0.1, "--clip-eps", help="epsilon used for GRPO clip"),
    kl_strength: float = typer.Option(1.0, "--kl", help="strength of the kl penalty to the reference policy"),
    # eval split
    eval_split: float = typer.Option(
        0.0, "--eval-split", help="portion of training samples to use for the eval dataset"
    ),
    output_dir: str | None = typer.Option(
        None, "--output-dir", help="Optional directory to use for saving the checkpoints"
    ),
):
    # initialize distributed
    init_distributed()

    # device setup
    # create training components
    train_ctx = TrainingContext(
        model_name=model_dir,
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
    )
    trainer = GRPOTrainer(train_ctx)

    dist.barrier()
    trainer.initialize()
    trainer.train()

    destroy_distributed_environment()


def stream_output(stream, prefix, file=sys.stdout):
    for line in iter(stream.readline, b""):
        print(f"[{prefix}] {line.decode().rstrip()}", file=file, flush=True)


@app.command()
def orchestrator(
    model_write_dir: str = "/dev/shm/mini-r1/current-model",
    checkpoint_dir: str | None = typer.Option(None, "--checkpoint-dir", help="Directory to save the checkpoint in"),
    train_gpus: int = typer.Option(6, "--train-gpus", help="Num GPUs to be used for training"),
    vllm_gpus: int = typer.Option(2, "--vllm-gpus", help="Num GPUs to be used for the vLLM server"),
    use_olmo: bool = typer.Option(
        False,
        "--use-olmo",
        help="Whether or not to use the Olmo 7B pretrained base (for replicating R1 zero)",
        is_flag=True,
    ),
    model_path: str | None = typer.Option(
        None, "--model-path", help="Path or name of the model to train with if not using olmo"
    ),
    # batch_size: int = typer.Option(288, "--batch-size", help="Batch size (for generation)"),
    # batch_size: int = typer.Option(54, "--batch-size", help="Batch size (for generation)"),
    # defaults are based on R1 Zero pipeline. R1 Zero used a batch size of 512, we have 6 GPUs so use 510
    batch_size: int = typer.Option(384, "--batch-size", help="Batch size (for generation)"),
    group_size: int = typer.Option(16, "--group-size", help="Rollout size (num rollouts to generate/batch)"),
    inner_batch_size: int = typer.Option(24, "--inner-batch-size", help="Inner batch size for the GRPO update step"),
    epochs: int = typer.Option(10, "--epochs", help="Number of epochs"),
    ref_update_frequency: int = typer.Option(
        400, "--ref-update-frequency", help="The frequency (in steps) in which the reference policy is updated."
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
    max_new_tokens: int = typer.Option(128, help="The maximum number of new tokens that the model can generate."),
    msl_jump_at=typer.Option(8200, help="Number of training epochs"),
    verbose_vllm: bool = typer.Option(False),
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
            "float16",
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
        ]
        if checkpoint_dir:
            train_cmd += ["--output-dir", checkpoint_dir]

        training_process = subprocess.Popen(
            train_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=train_env, bufsize=1
        )

        # stream stdout
        vllm_thread = threading.Thread(target=stream_output, args=(vllm_process.stdout, "VLLM"), daemon=True)
        training_thread = threading.Thread(target=stream_output, args=(training_process.stdout, "TRAIN"), daemon=True)

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
