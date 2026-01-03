import torch
import datasets
import random
from type_defs import (
    Problem,
    SamplingParams,
    Message,
    TokenSample,
    RolloutResult,
    Sample,
    Sample,
    TrainingComponents,
)
from torch.utils.data import DataLoader, Dataset

from instructlab.training.data_process import unmask_sample, configure_tokenizer, process_samples
from instructlab.training.type_definitions import ProcessedMessagesData
from transformers import PreTrainedTokenizer


def random_problems(seed: int = 42, num_problems: int = 20, min_num: int = 1, max_num: int = 100) -> list[Problem]:
    random.seed(seed)
    problems: list[Problem] = []
    for _ in range(num_problems):
        a, b = random.randint(min_num, max_num), random.randint(min_num, max_num)
        operation = random.choice(["add", "subtract"])
        if operation == "add":
            add_prompts = [
                f"What is the sum of {a} and {b}?",
                f"What is {a} plus {b}?",
                f"Add {a} and {b}.",
                f"Calculate {a} + {b}.",
                f"What do you get when you add {a} to {b}?",
                f"If you have {a} and add {b}, what is the total?",
            ]
            problem = random.choice(add_prompts)
            answer = a + b
        else:  # subtract
            subtract_prompts = [
                f"What is the difference of {a} and {b}?",
                f"What is {a} minus {b}?",
                f"Subtract {b} from {a}.",
                f"Calculate {a} - {b}.",
                f"What do you get when you subtract {b} from {a}?",
                f"If you have {a} and take away {b}, what is left?",
            ]
            problem = random.choice(subtract_prompts)
            answer = a - b
        problems.append(Problem(problem=problem, answer=answer, operation=operation))
    return problems


def generate_dataset(
    system_msg: str,
    num_problems: int = 20,
    min_num: int = -100,
    max_num: int = 100,
    seed: int = 42,
    # ) -> datasets.Dataset:
) -> datasets.Dataset:
    problems = random_problems(seed=seed, num_problems=num_problems, min_num=min_num, max_num=max_num)

    # Convert list of Problem objects to dataset
    problems_dict = [problem.model_dump() for problem in problems]
    dataset = datasets.Dataset.from_list(problems_dict)
    dataset = dataset.map(
        lambda x: {
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": x["problem"]},
            ]
        }
    )
    return dataset


class JsonlDataset(torch.utils.data.Dataset):
    """Dataset class for loading pre-tokenized input IDs from JSONL files."""

    def __init__(self, dataset: datasets.Dataset = None, data_path: str = None):
        """
        Initialize the dataset.

        Args:
            data_path: Path to the JSONL file containing input_ids
        """
        if dataset:
            self.dataset = dataset
        elif data_path:
            self.dataset = datasets.load_dataset("json", data_files=data_path, split="train")
        else:
            raise NotImplementedError()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Returns:
            dict: Dictionary containing 'input_ids' and other fields from the JSONL
        """
        item = self.dataset[idx]
        to_return = {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
            "len": len(item["input_ids"]),
        }
        if "advantage" in item:
            to_return["advantage"] = item["advantage"]
        if "logprobs" in item:
            to_return["logprobs"] = torch.tensor([it["logprob"] for it in item["logprobs"]], dtype=torch.float32)
            to_return["logprob_ids"] = torch.tensor([it["token"] for it in item["logprobs"]], dtype=torch.float32)
        return to_return


def collate_fn(batch: list[dict], pad_token_id: int):
    """
    batch is a list of dicts containing:
    - input_ids: tensor contining the input ids
    - labels: tensor contining the input ids
    """
    max_len = max(batch, key=lambda x: x["input_ids"].numel())["input_ids"].numel()
    # Pad all sequences to max_len
    input_ids_padded = []
    labels_padded = []
    attention_mask_padded = []
    num_tokens_in_batch = 0
    num_loss_tokens_in_batch = 0
    advantages = []
    logprob_ids_padded = []
    logprobs_padded = []
    logprobs_in_batch = []
    batch_grpo_mask = []

    for item in batch:
        seq_len = item["input_ids"].numel()
        num_tokens_in_batch += seq_len
        num_loss_tokens_in_batch += (item["labels"] != -100).sum().item()

        # Pad input_ids (typically with 0 or tokenizer.pad_token_id)
        full_input_seq = torch.full((max_len,), fill_value=pad_token_id, dtype=torch.long)
        full_labels = torch.full_like(full_input_seq, fill_value=-100, dtype=torch.long)
        full_attn_mask = torch.zeros_like(full_input_seq, dtype=torch.long)
        # full_attn_mask =

        # populate padded inputs with values from dataset
        idxs = torch.arange(0, seq_len)
        full_input_seq[idxs] = item["input_ids"]
        full_labels[idxs] = item["labels"]
        full_attn_mask[idxs] = 1  # should compute attention here
        assert full_attn_mask.equal((full_input_seq != pad_token_id).float())  # just verify this for now

        # update the batch items
        input_ids_padded += [full_input_seq]
        labels_padded += [full_labels]
        attention_mask_padded += [full_attn_mask]

        # also add the advantage to be aligned with the rest of the inputs
        if "advantage" not in item:
            continue

        # handle the GRPO pieces
        advantages.append(item["advantage"])

        # how far the logprobs (completed sequence) is from the beginning
        prompt_offset = seq_len - item["logprobs"].numel()
        completion_length = item["logprobs"].numel()

        # this should be constructed such that the default value will
        # have no effect on the compute graph
        full_logprobs = torch.ones_like(full_labels, dtype=torch.float32)
        # first ensure it's the same size
        assert full_logprobs[prompt_offset : prompt_offset + completion_length].numel() == item["logprobs"].numel()
        full_logprobs[prompt_offset : prompt_offset + completion_length] = item["logprobs"]

        # do the same for the logit ids
        full_logprob_ids = torch.full_like(full_labels, fill_value=pad_token_id)
        full_logprob_ids[prompt_offset : prompt_offset + completion_length] = item["logprob_ids"]
        grpo_mask = (full_logprob_ids.detach() != pad_token_id).detach()
        batch_grpo_mask += [grpo_mask]

        # now make sure to append all of these
        logprob_ids_padded += [full_logprob_ids]
        logprobs_padded += [full_logprobs]

        # count the number of tokens that we consider ourselves to actually be backproping on
        logprobs_in_batch.append(completion_length)

    final_item = {
        "input_ids": torch.stack(input_ids_padded).detach(),
        "labels": torch.stack(labels_padded).detach(),
        "attention_mask": torch.stack(attention_mask_padded).detach(),
        "num_tokens": num_tokens_in_batch,
        "num_loss_tokens": num_loss_tokens_in_batch,
        "num_sequences": len(batch),
    }

    if len(advantages) > 0:
        assert sum(logprobs_in_batch) == num_loss_tokens_in_batch, (
            f"number of loss tokens {num_loss_tokens_in_batch} doesn't match num logprobs {sum(logprobs_in_batch)}"
        )
        final_item.update(
            {
                "advantages": torch.tensor(advantages, dtype=torch.float32),
                "logprobs": torch.stack(logprobs_padded).detach(),
                "logprob_ids": torch.stack(logprob_ids_padded).detach(),
                "rollout_lens": torch.tensor(logprobs_in_batch, dtype=torch.long),
                "grpo_mask": torch.stack(batch_grpo_mask).detach(),
            }
        )

    return final_item


def samples_to_dataset(samples: list[Sample], tokenizer: PreTrainedTokenizer) -> datasets.Dataset:
    all_rollouts: list[RolloutResult] = []
    for s in samples:
        all_rollouts.extend(s.rollouts)
    dataset = datasets.Dataset.from_list([r.to_dataset_format() for r in all_rollouts], split="train")
    dataset = process_samples(dataset, tokenizer=tokenizer, num_cpu_procs=1)

    # HACK: this processing is specific to qwen and assumes a single-turn conversation
    # for each sample, we need to make sure that we do not include token ids beyond the
    # final eos_token

    def trim_input_ids_to_original_rollout(sample: dict):
        rollout_has_eos_tok = any(tokenizer.eos_token_id == tok["token"] for tok in sample["logprobs"])
        if not rollout_has_eos_tok:
            # do nothing
            return {
                # perform the shift here
                "input_ids": sample["input_ids"][:-1],
                "labels": sample["labels"][1:],
            }

        # otherwise we truncate everything beyond the eos token
        if (count := sample["input_ids"].count(tokenizer.eos_token_id)) < 3:
            raise ValueError(f"unexpected amount of eos token ids in input sequence, expected 3+, got {count}")

        # generate offset
        final_tok_idx = sample["input_ids"][::-1].index(tokenizer.eos_token_id)

        # we chop off the end of the input_ids sequence, so final eos token will no longer be present
        input_ids_offset_idx = -(final_tok_idx + 1)
        offset_input_ids = sample["input_ids"][
            :input_ids_offset_idx
        ]  # this drops the final eos token from input sequence
        if len(offset_input_ids) == 0:
            raise ValueError("tried to offset the final tokenizer index, but result is an empty sequence")

        # we chop off the end of labels
        if final_tok_idx > 0:
            offset_labels = sample["labels"][:-final_tok_idx]

        # then we shift labels forward to align with input ids
        offset_labels = offset_labels[1:]
        if len(offset_labels) == 0:
            raise ValueError("tried to offset the final tokenizer index, but result is an empty sequence")

        if len(offset_labels) != len(offset_input_ids):
            raise ValueError(
                f"concatentation error, offset labels arent equal to input ids: {len(offset_labels)} != {len(offset_input_ids)}"
            )

        # return {"trimmed_input_ids": offset_input_ids, "trimmed_labels": offset_labels}
        return {"input_ids": offset_input_ids, "labels": offset_labels}

    dataset = dataset.map(trim_input_ids_to_original_rollout, num_proc=1, desc="trimming input ids")

    # now we add the masked samples
    return dataset


def get_unmasked_sample(rollout: RolloutResult, tokenizer: PreTrainedTokenizer) -> ProcessedMessagesData:
    """
    wrapper around unmask_sample which drops all tokens after the final EOS token id
    """

    # process the sample
    result = unmask_sample({"messages": rollout.rollout_trace_to_json_list()}, tokenizer)

    last_idx = result["input_ids"][::-1].index(tokenizer.eos_token_id)
    truncated_input_ids = result["input_ids"][:-last_idx]
    truncated_labels = result["labels"][:-last_idx]

    return {
        "input_ids": truncated_input_ids,
        "labels": truncated_labels,
        "len": len(truncated_input_ids),
    }


def create_grpo_data_loader(dataset: datasets.Dataset, comps: TrainingComponents):
    from functools import partial

    _collate_fn = partial(collate_fn, pad_token_id=comps.tokenizer.pad_token_id)

    ds = JsonlDataset(dataset=dataset)
    train_loader = DataLoader(
        dataset=ds,
        collate_fn=_collate_fn,
        batch_size=comps.hyperparams.inner_batch_size,
        shuffle=True,
    )
    return train_loader
