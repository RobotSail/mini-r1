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
    TrainingContext,
)
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data import DistributedSampler
import torch.distributed as dist

from transformers import PreTrainedTokenizer
from IPython import embed


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
    system_msg: str | None = None,
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
            "messages": ([{"role": "system", "content": system_msg}] if system_msg else [])
            + [{"role": "user", "content": x["problem"]}],
        }
    )
    return dataset


class JsonlDataset(torch.utils.data.Dataset):
    """Dataset class for loading pre-tokenized input IDs from JSONL files."""

    def __init__(self, dataset: list[Sample], pad_token_id: int):
        """
        Initialize the dataset.

        Args:
            data_path: Path to the JSONL file containing input_ids
        """
        self.dataset = []
        # we must flatten the batch
        for sample in dataset:
            # these are the prompt input IDs
            prompt_input_ids = sample.input_ids
            for rollout in sample.rollouts:
                # skip
                if not rollout.logprobs:
                    continue

                # this should be good enough
                input_ids = prompt_input_ids[:] + rollout.token_ids[:]
                grpo_mask = [False] * len(prompt_input_ids) + [True] * len(rollout.token_ids[:])
                logprobs_seq = [1.0] * len(prompt_input_ids) + [lp.logprob for lp in rollout.logprobs]

                # now we shift so offset is correct and we predict causal
                grpo_mask = grpo_mask[1:]
                logprobs_seq = logprobs_seq[1:]
                input_ids = input_ids[:-1]
                num_logprobs = len(rollout.logprobs)

                # account for shift
                logprob_tokens: list[int] = [pad_token_id] * len(prompt_input_ids[1:]) + [
                    lp.token for lp in rollout.logprobs
                ]
                assert len(logprob_tokens) == len(logprobs_seq)

                self.dataset.append(
                    {
                        "input_ids": input_ids,
                        "logprobs": logprobs_seq,
                        "grpo_mask": grpo_mask,
                        "num_logprobs": num_logprobs,
                        "advantage": rollout.advantage,
                        "logprob_ids": logprob_tokens,
                    }
                )

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
            # "logprob_ids": torch.tensor(item["logprob_ids"], dtype=torch.long),
            "logprobs": torch.tensor(item["logprobs"], dtype=torch.float32),
            "grpo_mask": torch.tensor(item["grpo_mask"], dtype=torch.bool),
            "logprob_ids": torch.tensor(data=item["logprob_ids"], dtype=torch.long),
            # debug
            # "full_input_ids": torch.tensor(item["full_input_ids"], dtype=torch.long),
            # "full_logprob_ids": torch.tensor(item["full_logprob_ids"], dtype=torch.long),
            "advantage": item["advantage"],
            # "prompt_offset": item["prefix_len"],
            "num_logprobs": item["num_logprobs"],
            #            "input_ids": input_ids,
            # "logprob_ids": logprob_seq,
            # "grpo_mask": grpo_mask,
            # # adds all of these for debugging purposes
            # "full_input_ids": full_input_seq,
            # "full_logprob_ids": full_logprob_seq,
            # "logprobs": logprobs,
        }
        return to_return


def collate_fn(batch: list[dict], pad_token_id: int):
    """
    Here we collate using padding-free
    """
    num_tokens_in_batch = 0
    advantages = []
    input_ids = []
    logprobs = []
    grpo_mask = []
    position_ids: list[torch.Tensor] = []
    logprob_ids: list[torch.Tensor] = []

    # in order to average over the sequence length properly
    # we calculate the total batch size here
    grpo_scalars = []

    for i, item in enumerate(batch):
        last_item = i + 1 == len(batch)
        input_ids += [item["input_ids"]]
        logprob_ids += [item["logprob_ids"]]
        logprobs += [item["logprobs"]]
        grpo_mask += [item["grpo_mask"]]
        position_ids += [torch.arange(0, item["input_ids"].numel(), step=1)]

        # we will use these to divide the final logprobs sequence
        scalars = torch.masked_fill(
            torch.ones_like(item["input_ids"]),
            item["grpo_mask"],
            value=item["num_logprobs"],
        )
        advantages += [torch.full_like(item["grpo_mask"], item["advantage"], dtype=torch.float32)]
        grpo_scalars += [scalars]

        if not last_item:
            input_ids += [torch.tensor([pad_token_id], dtype=torch.long)]
            logprob_ids += [torch.tensor([pad_token_id], dtype=torch.long)]
            logprobs += [torch.tensor([1.0])]
            grpo_mask += [torch.tensor([0.0])]
            grpo_scalars += [torch.tensor([1])]
            advantages += [torch.tensor([0.0])]
            position_ids += [torch.tensor(data=[0.0])]

    final_item = {
        "input_ids": torch.cat(input_ids).detach(),
        "position_ids": torch.cat(position_ids).detach(),
        "advantages": torch.cat(advantages).detach(),
        "logprobs": torch.cat(logprobs).detach(),
        "logprob_ids": torch.cat(logprob_ids).detach(),
        "grpo_mask": torch.cat(grpo_mask).detach(),
        "scalars": torch.cat(grpo_scalars).detach(),
        "batch_size": len(batch),
    }
    return final_item


class ProblemDataset(torch.utils.data.Dataset):
    def __init__(self, ds: datasets.Dataset):
        self.dataset = ds

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Problem:
        item = self.dataset[index]
        return Problem(
            answer=item["answer"],
            operation=item["operation"],
            problem=item["problem"],
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @classmethod
    def from_jsonl(
        cls, path: str, eval_split: float = 0.0, seed: int = 67
    ) -> tuple["ProblemDataset", "ProblemDataset"]:
        train_dataset = datasets.load_dataset("json", data_files=path, split="train")
        eval_dataset = None

        if eval_split > 0:
            dataset_dict = train_dataset.train_test_split(test_size=eval_split, seed=seed)
            train_dataset = dataset_dict["train"]
            eval_dataset = dataset_dict["test"]

        return (ProblemDataset(train_dataset), ProblemDataset(eval_dataset) if eval_dataset else None)


def problem_collate_fn(batch: list[Problem]) -> list[Problem]:
    return batch


def create_distributed_data_loader(dataset, batch_size: int):
    # so we're gonna get a dataset in a format like
    # creates the distributed sampler
    train_sampler = DistributedSampler(
        dataset=dataset,
        drop_last=True,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),  # this needs the global rank, so this is correct,
        seed=67,
        shuffle=True,
    )
    batchsize = batch_size // dist.get_world_size()

    # we will fix eval later
    return DataLoader(dataset, sampler=train_sampler, batch_size=batchsize, collate_fn=problem_collate_fn)


def create_grpo_data_loader(
    ctx: TrainingContext,
    dataset: list[Sample],
):
    from functools import partial

    _collate_fn = partial(collate_fn, pad_token_id=ctx.tokenizer.pad_token_id)

    ds = JsonlDataset(dataset=dataset, pad_token_id=ctx.tokenizer.pad_token_id)
    sampler = DistributedSampler(
        ds,
        dist.get_world_size(),
        drop_last=True,
        shuffle=True,
    )
    train_loader = DataLoader(
        dataset=ds,
        sampler=sampler,
        collate_fn=_collate_fn,
        batch_size=ctx.hparams.inner_batch_size // dist.get_world_size(),
    )
    return train_loader
