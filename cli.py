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
import shutil
from IPython import embed

from instructlab.training.data_process import (
    process_messages_into_input_ids,
    configure_tokenizer,
)

from data_utils import generate_dataset, get_unmasked_sample, samples_to_dataset, create_grpo_data_loader
from utils import preview_tokenization, display_scorecard
from type_defs import (
    Problem,
    SamplingParams,
    TokenSample,
    RolloutResult,
    Sample,
    TrainingComponents,
    Hyperparameters,
)


# Regex pattern to match <answer>...</answer> tags
answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


app = Typer()


def send_chat_completion(
    prompt: str,
    system_prompt: str,
    model: str = "qwen/Qwen2-1.5B-Instruct",
    base_url: str = "http://localhost:8000/v1",
    temperature: float = 0.7,
    max_tokens: int = 512,
):
    """Send a chat completion request to vLLM server."""
    url = f"{base_url}/chat/completions"

    headers = {"Content-Type": "application/json"}

    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()

    return response.json()


def extract_answer(response: str) -> int:
    """Extract the numerical answer from the response using the <answer>...</answer> format."""
    match = answer_pattern.search(response)
    if match:
        try:
            return int(match.group(1).strip())
        except ValueError:
            raise ValueError(f"Could not extract answer from response: {response} (invalid integer)")
    raise ValueError(f"Could not extract answer from response: {response} (no <answer>...</answer> tags found)")


@app.command()
def generate_data(
    # system_msg: str,
    system_msg="You are a helpful math assistant. Always provide your final numerical answer inside of the <answer>...</answer> tags, e.g.: <answer>42</answer>",
    num_problems: int = 20,
    min_num: int = -100,
    max_num: int = 100,
    seed: int = 42,
    model_name: str = "qwen/Qwen2-1.5B-Instruct",
    output_dir: str = "generated_data",
    test_split: float = 0.0,
    max_seq_len: int = 8192,
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

    # write out
    train_raw_path = os.path.join(output_dir, "train_raw.jsonl")
    train.to_json(train_raw_path)

    if test:
        test_raw_path = os.path.join(output_dir, "test_raw.jsonl")
        test.to_json(test_raw_path)

    # here we create a parsed one
    tmp_dir = "/tmp/parsed_data"
    process_messages_into_input_ids(
        data_path=train_raw_path,
        data_output_path=os.path.join(tmp_dir),
        max_seq_len=max_seq_len,
        model_path=model_name,
        num_cpu_procs=4,
    )
    train_path = os.path.join(output_dir, "train.jsonl")
    shutil.move(os.path.join(tmp_dir, "data.jsonl"), os.path.abspath(train_path))
    typer.secho(
        f"✓ Processed training data, wrote results to {train_path}",
        fg=typer.colors.GREEN,
    )

    test_path = ""
    if test:
        # do the same for the test data
        process_messages_into_input_ids(
            data_path=train_raw_path,
            data_output_path=os.path.join(tmp_dir),
            max_seq_len=max_seq_len,
            model_path=model_name,
            num_cpu_procs=4,
        )
        test_path = os.path.join(output_dir, "test.jsonl")
        shutil.move(os.path.join(tmp_dir, "data.jsonl"), os.path.abspath(test_path))
        typer.secho(
            f"✓ Processed test data, wrote results to {test_path}",
            fg=typer.colors.GREEN,
        )

    typer.secho(f"✓ Generated {len(train)} training examples", fg=typer.colors.GREEN)
    if test:
        typer.secho(f"✓ Generated {len(test)} test examples", fg=typer.colors.GREEN)

    typer.secho(f"✓ Saved files to '{output_dir}'", fg=typer.colors.BLUE)
    typer.secho(f"✓ Tokenized messages data: '{train_path}'", fg=typer.colors.BLUE)
    typer.secho(f"✓ Raw training data: '{train_raw_path}'", fg=typer.colors.BLUE)
    if test:
        typer.secho(f"✓ Tokenized test data: '{test_path}'", fg=typer.colors.BLUE)
        typer.secho(f"✓ Raw test data: '{test_raw_path}'", fg=typer.colors.BLUE)


@torch.no_grad
def generate_rollouts(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    batch: dict[str, list[any]],
    batch_size: int,
    num_rollouts: int,
    sampling_params: SamplingParams,
) -> list[Sample]:
    model.eval()
    device = next(p.device for p in model.parameters())
    # here we need to create a set of rollouts for each prompt
    rollouts: list[Sample] = []
    for i in range(batch_size):
        # TODO: optimize this
        # Preview the messages for this batch item
        # if i == 0:  # Only preview the first item to avoid clutter
        #     typer.secho(f"\n[Batch {i}] Messages:", fg=typer.colors.BRIGHT_CYAN)
        #     for msg in batch["messages"][i]:
        #         typer.secho(
        #             f"  [{msg['role']}]: {msg['content']}", fg=typer.colors.CYAN
        #         )
        input_ids = tokenizer.apply_chat_template(
            conversation=batch["messages"][i],
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device=device)

        # now we sample
        outputs = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=sampling_params.max_new_tokens,
            num_return_sequences=num_rollouts,
            do_sample=True,
            temperature=sampling_params.temperature,
            top_k=sampling_params.top_k,
            top_p=sampling_params.top_p,
            repetition_penalty=sampling_params.repetition_penalty,
            # output_logits=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
        input_len = input_ids.numel()
        new_tokens = outputs.sequences[:, input_len:]

        # for each sample in the batch we append the generated responses as they're parsed back from the model
        # this should align with the rollout ordering that we get from the batch
        # TODO: vectorize logprob gathering

        # embed()

        # we recollect the sample by combining across the column dimension
        seed_sample = {k: v[i] for k, v in batch.items()}
        rollout_data: list[RolloutResult] = []

        # go through each sequence and grab the respective logprob
        # TODO: optimize this part
        for seq_idx, seq in enumerate(new_tokens.tolist()):
            logprobs: list[TokenSample] = []

            # stop processing after the model generated EOS token
            try:
                seq_end = seq.index(tokenizer.eos_token_id) + 1
            except ValueError:
                # fallback to full sequence
                seq_end = len(seq)

            # next, we just need to select the probs for our specific tokens
            processed_logits = torch.stack([t[seq_idx] for t in outputs.scores[:seq_end]])
            processed_probs = processed_logits.softmax(dim=-1)
            index = torch.tensor(seq[:seq_end], dtype=torch.long, device=processed_logits.device)
            index = index.unsqueeze(-1)  # extend from (T,) into (T, 1)
            probs = processed_probs.gather(dim=-1, index=index)
            probs = probs.squeeze(-1)  # (T, 1) --> (T,)
            index = index.squeeze(-1)  # (T, 1) --> (T,)

            for tok, prob in zip(index.tolist(), probs.tolist()):
                logprobs.append(
                    TokenSample(
                        token=tok,
                        logprob=prob,
                    )
                )

            # for rollout_idx in range(num_rollouts):
            #     logprobs: list[TokenSample] = []
            #     for token_idx, new_tok in zip(
            #         range(len(new_tokens[rollout_idx])),
            #         new_tokens[rollout_idx],
            #     ):
            #         # since generation pads out the responses, the padding tokens were not sampled by the policy
            #         # and therefore they must be skipped
            #         if new_tok == tokenizer.pad_token_id:
            #             break

            #         logits = outputs.logits[token_idx][rollout_idx]  # returns the logit distribution for a given rollout
            #         # we need to collect 2 things:
            #         # 1. the sampled token
            #         # 2. the logprob of the sampled token
            #         logprob = logits.softmax(-1)[new_tok]
            #         logprobs.append(
            #             TokenSample(
            #                 token=new_tok.item(),
            #                 logprob=logprob.item(),
            #                 logit=logits[new_tok].item(),
            #             )
            #         )

            # here we append the rollout data
            policy_response = tokenizer.decode(new_tokens[seq_idx], skip_special_tokens=True)
            rollout_data.append(
                RolloutResult(
                    logprobs=logprobs,
                    response=policy_response,
                    seed_messages=seed_sample["messages"],
                )
            )

        rollouts.append(
            Sample(
                problem=Problem(
                    answer=seed_sample["answer"],
                    operation=seed_sample["operation"],
                    problem=seed_sample["problem"],
                ),
                rollouts=rollout_data,
            )
        )

    # empty cache
    torch.cuda.empty_cache()
    return rollouts


def grade_samples(samples: list[Sample]):
    """
    Given a batch of samples, calculates the advantage for each one.
    Modifies objects in place.
    """
    for sample in samples:
        for rollout in sample.rollouts:
            # Defaults; if we cannot parse then it is not correct. If we can parse, it is not necessarily correct.
            rollout.is_parsable = False
            rollout.is_correct = False

            # try to parse the response:
            try:
                answer = extract_answer(rollout.response)
                rollout.is_parsable = True
            except Exception as e:
                typer.secho(
                    f"failed to parse response '{rollout.response}': {e}",
                    fg=typer.colors.RED,
                )
            else:
                rollout.is_correct = sample.problem.answer == answer


def calculate_reward(samples: list[Sample]):
    """
    Function containing reward calculation logic
    """
    for sample in samples:
        for rollout in sample.rollouts:
            # reset it here just for good measure
            rollout.reward = 0
            if rollout.is_parsable:
                # okay, we WANT the model to produce more answers like this
                # but we don't want to overweight this or give sparse rewards
                # so we will assign a reward here of +1
                rollout.reward += 1
            if rollout.is_correct:
                rollout.reward += 5  # huge reward for getting it right


# i dont think we even have tensors flowing through this function but you
# can never be too sure.
@torch.no_grad
def calculate_advantage(samples: list[Sample]):
    r"""
    This is the fun part, we have to implement the GRPO-style
    advantage calculation. Basically we take each set of rollouts as a single
    group and we calculate a group-level advantage as a workaround for
    not being able to calculate RTG or step-level advantage as in vanilla REINFORCE.

    Formula looks like this:

    $$
    A_i = \frac{r_i - \mean(r)}{\std(r) + \epsilon}
    $$
    """
    eps = 1e-8
    for sample in samples:
        avg = sum(r.reward for r in sample.rollouts) / len(sample.rollouts)
        var = sum((r.reward - avg) ** 2 for r in sample.rollouts) / len(sample.rollouts)
        std = var**0.5

        # if std < eps (because all rewards are equal) we use the std trick
        # of setting group advantage to 0
        enable_std_trick = std < eps

        # GRPO simple advantage
        for rollout in sample.rollouts:
            if enable_std_trick:
                rollout.advantage = 0.0
            else:
                rollout.advantage = (rollout.reward - avg) / (std + eps)


def train_policy_on_rollouts(samples: list[Sample], comps: TrainingComponents):
    comps.model.train()

    # we need to create a dataset here
    # configure tokenizer first
    # create the 'dataset'
    dataset = samples_to_dataset(samples, comps.train_tokenizer)
    data_loader = create_grpo_data_loader(dataset, comps)

    # so then we need to create a dataset from the rollouts
    # our dataset needs:
    #  1. messages
    #  2. advantage for rollout
    #  3. logprobs

    # now we train
    for _ in range(comps.hyperparams.inner_epochs):
        # generate the random set

        for batch in data_loader:
            """
            minibatch here has columns input_ids, labels, advantage. it is indexed column-first and row-second
            """

            print("start of batch :)")
            embed()

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
            input_ids = batch["input_ids"].to(comps.device)
            advantages = batch["advantages"].to(comps.device)
            ref_logprobs = batch["logprobs"].to(comps.device)
            ref_logprob_ids = batch["logprob_ids"].to(comps.device)
            rollout_lens = batch["rollout_lens"].to(comps.device)
            attn_mask = batch["attention_mask"].to(comps.device)
            grpo_logit_mask = batch["grpo_mask"].to(comps.device)

            # we're not calculating cross-entropy against static labels, so no label inputs
            # are needed here
            outputs = comps.model(input_ids=input_ids, attention_mask=attn_mask)
            new_logits = outputs.logits

            # account for sampling temperature
            if comps.sampling_params.temperature > 0:
                new_logits /= comps.sampling_params.temperature

            # we need the new logprobs

            # prepare for GRPO loss calculation, pluck out the logits that we're going to work with
            # 1. create the indices
            assert comps.tokenizer.pad_token_id is not None and comps.tokenizer.pad_token_id > 0
            gather_indices = ref_logprob_ids.clone().unsqueeze(
                -1
            )  # (B, T) --> (B, T, 1) # add a new dimension to index logits

            # 2. Calculate the latest logprobs
            new_probs: torch.Tensor = new_logits.softmax(dim=-1)

            # 3. Extract the probs from the original rollout
            new_rollout_probs = new_probs.gather(dim=-1, index=gather_indices)
            # (B, T, V) --> (B, T, 1)
            new_rollout_probs = new_rollout_probs.squeeze(-1)  # (B, T, 1) --> (B, T)
            assert len(new_rollout_probs.shape) == 2

            # 4. Compute  te importance ratio
            # here's where we need to actually compute the loss. first thing we need is to calculate
            # the importance ratio:
            # \rho(\theta_0)=\exp(\log p_{\theta_0}-\log p_{\text{old}})
            assert ref_logprobs.shape == new_rollout_probs.shape
            importance_ratio: torch.Tensor = (new_rollout_probs.log() - ref_logprobs.log()).exp()

            # For debugging, this should be true on the first time
            # try:
            #     assert importance_ratio.allclose(torch.ones_like(importance_ratio), rtol=1e-4)
            # except AssertionError as e:
            #     print(f"got assertion error: {e}")
            #     embed()

            # 5. Next we calculate the clipped surrogate objective
            # adjust advantage so it can broadcast cleanly
            advantages = advantages.unsqueeze(-1)  # (B,) --> (B, 1)

            # (B, 1) * (B, T)
            unclipped = advantages * importance_ratio
            clipped = advantages * importance_ratio.clamp(1 - comps.hyperparams.eps, 1 + comps.hyperparams.eps)
            clipped_surrogate = torch.minimum(unclipped, clipped)

            # 6. Next, we need to calculate the approximate KL penalty to the old policy
            # KL(policy_new || policy_old) ~= policy_old/policy_new - log(policy_old/policy_new) - 1
            dkl_approx = (
                (ref_logprobs.log() - new_rollout_probs.log()).exp()
                - (ref_logprobs.log() - new_rollout_probs.log())
                - 1
            )

            # 7. Compute per-token loss L_GRPO = L_clip - L_kl
            # compute the per-sample loss (loss for all samples is independent at this point)
            assert dkl_approx.shape == clipped_surrogate.shape, f"{dkl_approx.shape=} != {clipped_surrogate.shape=}"
            per_token_loss = clipped_surrogate - comps.hyperparams.kl_penalty_strength * dkl_approx

            # 8. Mask out all invalid logprobs that aren't from the GRPO rollouts
            grpo_token_loss = per_token_loss * grpo_logit_mask.float()

            # --- loss aggregation ---
            # 9. Next, we average each batch by the **sequence length**, then we average by the group-size
            # Warning: sequence-length averaging assigns greater weight to shorter sequences than longer ones
            # but in our case this is fine, since we only care about an objective reward
            # (B,) --> (B, 1)
            grpo_sequence_loss = grpo_token_loss.sum(dim=-1) / rollout_lens.float()
            assert len(grpo_sequence_loss.shape) == 1
            grpo_loss = grpo_sequence_loss.mean()  # group average
            # backprop
            grpo_loss.backward()

            # we optimize the model
            gradnorm = torch.nn.utils.clip_grad_norm_(comps.model.parameters(), 1.0)
            comps.optimizer.step()
            comps.optimizer.zero_grad()

            print(f"reached the end of the optimize step: {gradnorm=}")

            embed()
            import sys

            sys.exit(0)


@app.command()
def train(
    # dataset parameters, we'll eventually move these to a data generation command
    data_path: str = typer.Option(..., help="Path to the training data"),
    seed: int = typer.Option(67, help="Random seed"),
    num_problems: int = typer.Option(20, help="Number of problems"),
    min_num: int = typer.Option(-100, help="Minimum number for problems"),
    max_num: int = typer.Option(100, help="Maximum number for problems"),
    # model
    model_name: str = typer.Option("qwen/Qwen2-1.5B-Instruct", help="Model name or path"),
    # training params
    epochs: int = typer.Option(1, help="Number of training epochs"),
    max_new_tokens: int = typer.Option(128, help="The maximum number of new tokens that the model can generate."),
    max_seq_len: int = typer.Option(
        8192,
        "--msl",
        "--max-seq-len",
        help="maximum length of the sequences that we work with",
    ),
    # adamw parms
    lr: float = typer.Option(1e-5, "--lr", help="Learning rate"),
    beta1: float = typer.Option(0.9, help="Adam beta1 parameter"),
    beta2: float = typer.Option(0.95, help="Adam beta2 parameter"),
    wd: float = typer.Option(0.0, "--wd", help="Weight decay"),
    # temporary
    training_gpu: int = typer.Option(1, help="GPU device for training"),
    # GRPO params
    inner_epochs: int = typer.Option(1, help="Number of passes on inner generation"),
    inner_batch_size: int = typer.Option(4, "--inner-batch-size", help="Batch size during the GRPO inner loop."),
    batch_size: int = typer.Option(1, "-B", "--batch-size", help="Batch size for training"),
    num_rollouts: int = typer.Option(1, "-G", "--generations", help="Number of rollouts"),
    temperature: float = typer.Option(0.7, "-t", "--temp", help="sampling temperature"),
    eps: float = typer.Option(0.1, "--eps", help="epsilon used for GRPO clip"),
    kl_strength: float = typer.Option(1.0, "--kl", help="strength of the kl penalty to the reference policy"),
):
    # load the raw dataset
    # train_dataset = JsonlDataset(data_path)
    train_dataset = datasets.load_dataset("json", data_files=data_path, split="train")

    # devices
    train_device = torch.device("cuda", training_gpu)

    # initialize model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=train_device,  # attn_implementation="flash_attention_2"
    )
    tokenizer: Qwen2Tokenizer = AutoTokenizer.from_pretrained(model_name)

    # align tokenizer and tokens
    if tokenizer.pad_token_id and not model.config.pad_token_id:
        typer.secho(
            f"model '{model_name}' doesn't have a pad_token_id, setting it to {tokenizer.pad_token_id}",
            fg=typer.colors.BRIGHT_BLUE,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=wd)

    training_comps = TrainingComponents(
        optimizer=optimizer,
        model=model,
        tokenizer=tokenizer,
        device=train_device,
        hyperparams=Hyperparameters(
            lr=lr,
            model_name=model_name,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            num_rollouts=num_rollouts,
            epochs=epochs,
            inner_epochs=inner_epochs,
            inner_batch_size=inner_batch_size,
        ),
        train_tokenizer=configure_tokenizer(model_name),
        sampling_params=SamplingParams(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            max_tokens=max_seq_len,
            top_p=1.0,
            top_k=0.0,
            repetition_penalty=1.0,
        ),
    )

    preview_tokenization(train_dataset, tokenizer)
    # now we iterate
    for epoch in range(epochs):
        minibatches: list[Sample] = []
        for batch in train_dataset.shuffle().iter(batch_size):
            # for batch in train_loader:
            #     input_ids = batch["input_ids"].to(device=inference_device)
            #     attention_mask = batch["attention_mask"].to(device=inference_device)
            #     from IPython import embed

            # here we need to create a set of rollouts for each prompt
            rollouts = generate_rollouts(
                model,
                tokenizer,
                batch,
                batch_size,
                num_rollouts,
                sampling_params=training_comps.sampling_params,
            )

            # now that we have rollouts, we must calculate advantage
            grade_samples(rollouts)
            calculate_reward(rollouts)
            calculate_advantage(rollouts)

            # \rho(\theta_0)=\exp(\log p_{\theta_0}-\log p_{\text{old}})
            minibatches.extend(rollouts)

        # now that we've generated G rollouts for our B groups of prompts,
        # we convert this into a dataset and update the trainable policy on it
        train_policy_on_rollouts(
            minibatches,
            training_comps,
        )

        # Calculate and display epoch scorecard
        display_scorecard(minibatches, epoch, epochs)


# @app.command()
# def main(
#     seed: int = 42,
#     num_problems: int = 20,
#     min_num: int = 1,
#     max_num: int = 100,
#     num_samples: int = 1,
# ):
#     problems = random_problems(seed, num_problems, min_num, max_num)
#     system_prompt = "You are a helpful math assistant. Always provide your final numerical answer using the format <answer>42</answer>"
#     samples: list[Sample] = []
#     for problem in problems:
#         sample = Sample(problem=problem)
#         for _ in range(num_samples):
#             # get model answer
#             sample.results.append(sample_result(problem, system_prompt))
#         samples.append(sample)

#     for sample in samples:
#         typer.echo(f"\nProblem: {sample.problem.problem}")
#         typer.echo(f"Expected Answer: {sample.problem.answer}")

#         # Count correct and incorrect answers
#         parsed_results = [r for r in sample.results if r.parsed_answer]
#         if not parsed_results:
#             typer.echo("No successfully parsed answers", color=typer.colors.YELLOW)
#             continue

#         correct_count = sum(1 for r in parsed_results if r.correct)
#         incorrect_count = len(parsed_results) - correct_count
#         total = len(parsed_results)

#         correct_pct = (correct_count / total * 100) if total > 0 else 0
#         incorrect_pct = (incorrect_count / total * 100) if total > 0 else 0

#         typer.echo(f"Correct: {correct_count}/{total} ({correct_pct:.1f}%)")
#         typer.echo(f"Incorrect: {incorrect_count}/{total} ({incorrect_pct:.1f}%)")

#         # Majority voting
#         if correct_count > incorrect_count:
#             typer.secho("Majority Vote: CORRECT", fg=typer.colors.GREEN)
#         else:
#             typer.secho("Majority Vote: INCORRECT", fg=typer.colors.RED)


if __name__ == "__main__":
    app()
