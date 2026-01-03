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

from instructlab.training.data_process import (
    configure_tokenizer,
)

from data_utils import generate_dataset, dataset_from_groups, create_grpo_data_loader
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


def parse_number(text: str) -> int | float:
    """Try to parse a string into a number which is assumed to been inside of the answer tags using the <answer>...</answer> format."""
    try:
        if not any(c.isdigit() for c in text):
            raise ValueError(f"No digits found in answer tags: {text}.")
        if "." in text:
            return float(text)
        else:
            return int(text)

    except ValueError as ve:
        raise ValueError(f"Could not extract answer from field: {text} (invalid integer): {ve}.")


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

    # write out training data
    train_path = os.path.join(output_dir, "train.jsonl")
    train.to_json(train_path)
    typer.secho(
        f"✓ Generated {len(train)} training examples",
        fg=typer.colors.GREEN,
    )
    typer.secho(
        f"✓ Saved training data to '{train_path}'",
        fg=typer.colors.BLUE,
    )

    # write out test data if it exists
    if test:
        test_path = os.path.join(output_dir, "test.jsonl")
        test.to_json(test_path)
        typer.secho(
            f"✓ Generated {len(test)} test examples",
            fg=typer.colors.GREEN,
        )
        typer.secho(
            f"✓ Saved test data to '{test_path}'",
            fg=typer.colors.BLUE,
        )


@torch.no_grad
def generate_rollouts(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    batch: dict[str, list[any]],
    batch_size: int,
    group_size: int,
    sampling_params: SamplingParams,
    show_tqdm=False,
) -> list[Sample]:
    model.eval()
    device = next(p.device for p in model.parameters())
    # here we need to create a set of rollouts for each prompt
    groups: list[Sample] = []

    iterator = range(batch_size)
    if show_tqdm:
        iterator = tqdm(iterator, desc="Generating rollouts")

    for i in iterator:
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
            num_return_sequences=group_size,
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
        problem = Problem(
            answer=seed_sample["answer"],
            operation=seed_sample["operation"],
            problem=seed_sample["problem"],
        )

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
            ref_logprobs = processed_logits.log_softmax(dim=-1)
            index = torch.tensor(seq[:seq_end], dtype=torch.long, device=processed_logits.device)
            index = index.unsqueeze(-1)  # extend from (T,) into (T, 1)
            probs = ref_logprobs.gather(dim=-1, index=index)
            probs = probs.squeeze(-1)  # (T, 1) --> (T,)
            index = index.squeeze(-1)  # (T, 1) --> (T,)

            for tok, prob in zip(index.tolist(), probs.tolist()):
                logprobs.append(
                    TokenSample(
                        token=tok,
                        logprob=prob,
                    )
                )

            # here we append the rollout data
            policy_response = tokenizer.decode(new_tokens[seq_idx], skip_special_tokens=True)
            rollout_data.append(
                RolloutResult(
                    logprobs=logprobs,
                    response=policy_response,
                    seed_messages=seed_sample["messages"],
                )
            )

        assert input_ids.ndim > 1
        groups.append(
            Sample(
                problem=problem,
                rollouts=rollout_data,
                input_ids=input_ids.tolist()[0],  # record the input ids so we can reuse them later
            )
        )

    for group in groups:
        grade_groups(group)
        calculate_advantage(group)

    # empty cache
    torch.cuda.empty_cache()
    return groups


@torch.no_grad
def grade_groups(group: Sample):
    """
    Given a batch of samples, calculates the advantage for each one.
    Modifies objects in place.
    """
    for rollout in group.rollouts:
        # Defaults; if we cannot parse then it is not correct. If we can parse, it is not necessarily correct.
        rollout.is_parsable = False
        rollout.is_correct = False

        # reset it here just for good measure
        rollout.reward = 0

        # check if the response has any answers at all
        matches = answer_pattern.findall(rollout.response)

        # we only want 1 of these
        parsed_nums: list[int | float] = []  #
        for match in matches:  # this is already bad
            try:
                answer = parse_number(match)
            except Exception as e:
                print(f"failed to parse text from answer tags: {e}")
            else:
                parsed_nums.append(answer)

        if len(parsed_nums) == 1:
            answer = parsed_nums[0]
            rollout.is_parsable = True
            rollout.is_correct = group.problem.answer == answer
        if len(parsed_nums) > 1:
            rollout.reward -= 0.1
        elif rollout.is_parsable:
            # okay, we WANT the model to produce more answers like this
            # but we don't want to overweight this or give sparse rewards
            # so we will assign a reward here of +1
            rollout.reward += 0.1

        if rollout.is_correct:
            rollout.reward += 1  # huge reward for getting it right


# i dont think we even have tensors flowing through this function but you
# can never be too sure.
@torch.no_grad
def calculate_advantage(group: Sample):
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
    avg = sum(r.reward for r in group.rollouts) / len(group.rollouts)
    var = sum((r.reward - avg) ** 2 for r in group.rollouts) / len(group.rollouts)
    std = var**0.5

    # if std < eps (because all rewards are equal) we use the std trick
    # of setting group advantage to 0
    enable_std_trick = std < eps

    # GRPO simple advantage
    for rollout in group.rollouts:
        if enable_std_trick:
            rollout.advantage = 0.0
        else:
            rollout.advantage = (rollout.reward - avg) / (std + eps)


@torch.no_grad
def eval_model(eval_dataset: datasets.Dataset, comps: TrainingComponents):
    comps.model.eval()

    # we generate all the rollouts
    eval_data = eval_dataset.batch(eval_dataset.num_rows)
    pass_at = [
        1,
    ]  #  3,#  5, 10]
    results = []

    for npass in pass_at:
        samples = generate_rollouts(
            comps.model,
            comps.tokenizer,
            batch=next(iter(eval_data)),
            batch_size=eval_dataset.num_rows,
            group_size=npass,
            sampling_params=comps.sampling_params,
            show_tqdm=True,
        )

        # now we go and determine the passing rate
        percent_scores = []
        for sample in samples:
            passing_rate = sum(1 if r.is_correct else 0 for r in sample.rollouts) / len(sample.rollouts)
            percent_scores.append(passing_rate)
        # Calculate statistics
        percent_above_50 = sum(1 if score > 0.5 else 0 for score in percent_scores) / len(percent_scores) * 100
        percent_at_100 = sum(1 if score == 1.0 else 0 for score in percent_scores) / len(percent_scores) * 100

        results.append((npass, percent_above_50, percent_at_100))

    # Print all results at the end
    typer.secho("\n=== Evaluation Scorecard ===", fg=typer.colors.BRIGHT_MAGENTA)
    typer.secho(f"Total samples evaluated: {len(samples)}", fg=typer.colors.BRIGHT_BLUE)
    for npass, percent_above_50, percent_at_100 in results:
        typer.secho(
            f"Pass@{npass}: {percent_above_50:.1f}% above 50% | {percent_at_100:.1f}% at 100% (across {len(samples)} samples with {npass} rollout(s) each)",
            fg=typer.colors.CYAN,
        )


def train_policy_on_rollouts(samples: list[Sample], comps: TrainingComponents):
    comps.model.train()

    # we need to create a dataset here
    # configure tokenizer first
    # create the 'dataset'
    dataset = dataset_from_groups(samples, comps.train_tokenizer)

    # so then we need to create a dataset from the rollouts
    # our dataset needs:
    #  1. messages
    #  2. advantage for rollout
    #  3. logprobs

    # now we train
    for epoch in range(comps.hyperparams.inner_epochs):
        # generate the random set

        data_loader = create_grpo_data_loader(dataset, comps)
        for batch in data_loader:
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
            input_ids = batch["input_ids"].to(comps.device)
            advantages = batch["advantages"].to(comps.device)
            old_logprobs = batch["logprobs"].to(comps.device)
            old_logprob_ids = batch["logprob_ids"].to(comps.device)
            rollout_lens = batch["rollout_lens"].to(comps.device)
            attn_mask = batch["attention_mask"].to(comps.device)
            grpo_logit_mask = batch["grpo_mask"].to(comps.device)

            # we're not calculating cross-entropy against static labels, so no label inputs
            # are needed here
            new_outputs = comps.model(input_ids=input_ids, attention_mask=attn_mask)
            new_logits = new_outputs.logits

            # account for sampling temperature
            if comps.sampling_params.temperature > 0:
                new_logits /= comps.sampling_params.temperature

            # now let's get the reference logits, but we really want to make sure that we don't
            # backprop on this model
            with torch.no_grad():
                ref_outputs = comps.ref_model(input_ids, attention_mask=attn_mask)
                ref_logits = ref_outputs.logits
                if comps.sampling_params.temperature > 0:
                    ref_logits /= comps.sampling_params.temperature

            # we need the new logprobs

            # prepare for GRPO loss calculation, pluck out the logits that we're going to work with
            # 1. create the indices
            gather_indices = old_logprob_ids.clone().unsqueeze(
                -1
            )  # (B, T) --> (B, T, 1) # add a new dimension to index logits

            # 2. Calculate the latest logprobs
            # more efficient technique
            new_gathered_logits = new_logits.gather(dim=-1, index=gather_indices)
            new_logsumexp = new_logits.logsumexp(dim=-1, keepdim=True)  # (B, T, 1)
            new_logprobs = (new_gathered_logits - new_logsumexp).squeeze(-1)

            # new_logprobs: torch.Tensor = new_logits.log_softmax(dim=-1)
            # # (B, T, V) --> (B, T, 1)
            # new_logprobs = new_logprobs.gather(dim=-1, index=gather_indices)
            # # (B, T, 1) --> (B, T)
            # new_logprobs = new_logprobs.squeeze(-1)
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

            # 5. Next we calculate the clipped surrogate objective
            # adjust advantage so it can broadcast cleanly
            advantages = advantages.unsqueeze(-1)  # (B,) --> (B, 1)

            # (B, 1) * (B, T)
            unclipped = advantages * importance_ratio
            clipped = advantages * importance_ratio.clamp(1 - comps.hyperparams.eps, 1 + comps.hyperparams.eps)
            clipped_surrogate = torch.minimum(unclipped, clipped)

            # 6. Next, we need to calculate the approximate KL penalty to the ref policy
            # KL(policy_new || policy_ref) ~= policy_ref/policy_new - log(policyref/policy_new) - 1
            dkl_approx = (ref_logprobs - new_logprobs).exp() - (ref_logprobs - new_logprobs) - 1

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
            grpo_loss = -grpo_sequence_loss.mean()  # group average

            # backprop
            grpo_loss.backward()

            # we optimize the model
            gradnorm = torch.nn.utils.clip_grad_norm_(comps.model.parameters(), 1.0)

            # take an optimization step
            comps.optimizer.step()
            comps.optimizer.zero_grad()

            # log metrics
            typer.secho(
                # f"[Epoch {epoch + 1}/{comps.hyperparams.epochs}] "
                f"Inner Epoch {epoch + 1}/{comps.hyperparams.inner_epochs} | "
                f"Loss: {grpo_loss.item():.4f} | "
                f"Grad Norm: {gradnorm.item():.4f}",
                fg=typer.colors.YELLOW,
            )


@app.command()
def train(
    # dataset parameters, we'll eventually move these to a data generation command
    train_path: str = typer.Option(..., "--train-path", help="Path to the training data"),
    eval_path: str = typer.Option(None, "--eval-path", help="Path to the training data"),
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
    batch_size: int = typer.Option(
        1, "-B", "--batch-size", help="Number of prompts to batch together when generating GRPO rollouts."
    ),
    group_size: int = typer.Option(
        1, "-G", "--group-size", help="Group size / number of rollouts to generate from a single prompt"
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
    # load the raw dataset
    # train_dataset = JsonlDataset(data_path)
    train_dataset = datasets.load_dataset("json", data_files=train_path, split="train")
    eval_dataset = None

    if eval_split > 0:
        dataset_dict = train_dataset.train_test_split(test_size=eval_split, seed=seed)
        train_dataset = dataset_dict["train"]
        eval_dataset = dataset_dict["test"]

    # Print dataset statistics
    typer.secho(f"\n✓ Loaded {len(train_dataset)} training samples", fg=typer.colors.GREEN)
    if eval_dataset:
        typer.secho(f"✓ Loaded {len(eval_dataset)} evaluation samples", fg=typer.colors.GREEN)

    # devices
    train_device = torch.device("cuda", training_gpu)

    # initialize model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=train_device,  # attn_implementation="flash_attention_2"
    )

    # this is a frozen mdel which we do not update
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=train_device,  # attn_implementation="flash_attention_2"
    )
    ref_model.eval()
    ref_model.requires_grad_(False)
    tokenizer: Qwen2Tokenizer = AutoTokenizer.from_pretrained(model_name)

    # align tokenizer and tokens
    for m in [model, ref_model]:
        if tokenizer.pad_token_id and not m.config.pad_token_id:
            m.config.pad_token_id = tokenizer.pad_token_id
            typer.secho(
                f"model '{model_name}' doesn't have a pad_token_id, setting it to {tokenizer.pad_token_id}",
                fg=typer.colors.BRIGHT_BLUE,
            )

    # create training components
    optimizer = AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=wd)
    training_comps = TrainingComponents(
        optimizer=optimizer,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        device=train_device,
        hyperparams=Hyperparameters(
            lr=lr,
            model_name=model_name,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            group_size=group_size,
            epochs=epochs,
            inner_epochs=inner_epochs,
            inner_batch_size=inner_batch_size,
            eps=clip_eps,
            kl_penalty_strength=kl_strength,
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
        output_dir=output_dir,
    )

    # check if we need to write into output dir
    if output_dir is not None and not training_comps.valid_save_dir():
        typer.secho(
            f"Error: Cannot write to output directory '{output_dir}'",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    preview_tokenization(train_dataset, tokenizer)
    if eval_dataset is not None and len(eval_dataset) > 0:
        eval_model(eval_dataset, training_comps)
    # now we iterate
    for epoch in range(epochs):
        minibatches: list[Sample] = []
        pbar = tqdm(
            train_dataset.shuffle().iter(batch_size),
            desc=f"Epoch {epoch + 1}/{epochs}",
            total=len(train_dataset) // batch_size,
        )
        for batch in pbar:
            # here we need to create a set of rollouts for each prompt
            rollouts = generate_rollouts(
                model,
                tokenizer,
                batch,
                training_comps.hyperparams.batch_size,
                training_comps.hyperparams.group_size,
                sampling_params=training_comps.sampling_params,
            )

            # Calculate average reward for this batch
            total_rewards = sum(rollout.reward for sample in rollouts for rollout in sample.rollouts)
            total_rollouts = sum(len(sample.rollouts) for sample in rollouts)
            avg_reward = total_rewards / total_rollouts if total_rollouts > 0 else 0.0

            # Update tqdm postfix with average reward
            pbar.set_postfix({"avg_reward": f"{avg_reward:.4f}"})

            # now that we've generated G rollouts for our B groups of prompts,
            # we convert this into a dataset and update the trainable policy on it
            train_policy_on_rollouts(
                rollouts,
                training_comps,
            )
            minibatches.extend(rollouts)

        # Calculate and display epoch scorecard
        display_scorecard(minibatches, epoch, epochs)

        # evaluate the model
        if eval_dataset is not None and len(eval_dataset) > 0:
            eval_model(eval_dataset, training_comps)

        # save a checkpoint for the current epoch
        training_comps.save_checkpoint(epoch)


if __name__ == "__main__":
    app()
