import torch
import datasets
import random

# from distributed_packer import pack_for_distributed_training
from type_defs import Problem, Sample, JsonlDatasetEntry
import re
import torch.distributed as dist
from torch.utils.data import Sampler, DataLoader, BatchSampler
import numpy as np
from src.batch_packer import batch_lengths_to_minibatches_lpt
import functools


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

    dataset: list[JsonlDatasetEntry]
    pad_token_id: int

    def _create_empty_entry(self, delimiter: bool = False) -> JsonlDatasetEntry:
        """
        Create an empty sample that can be used for padding samples
        """
        return {
            "input_ids": [self.pad_token_id],
            "logprobs": [1.0],
            "grpo_mask": [False],
            "num_logprobs": 0,
            "advantage": 0.0,
            "logprob_ids": [self.pad_token_id],
            "is_delimiter": delimiter,
            "is_padding": not delimiter,
        }

    def __init__(self, dataset: list[Sample], pad_token_id: int):
        """
        Initialize the dataset.

        Args:
            data_path: Path to the JSONL file containing input_ids
        """

        self.dataset = []
        self.pad_token_id = pad_token_id

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
                        "is_delimiter": False,
                        "is_padding": False,
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
        # handle padding indices
        is_padding = idx == -1
        is_delimiter = idx == -67
        if is_padding:
            item = self._create_empty_entry()
        elif is_delimiter:
            item = self._create_empty_entry(delimiter=True)
        else:
            item = self.dataset[idx]

        to_return = {
            "is_padding": is_padding,
            "is_delimiter": is_delimiter,
            "seq_len": len(item["input_ids"]),
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


class OlegDistributedSampler(BatchSampler):
    def __init__(
        self, dataset: JsonlDataset, max_tokens_per_gpu: int, global_batch_size: int, pad_token_id: int, seed: int = 67
    ):
        self.dataset = dataset
        self.max_tokens_per_gpu = max_tokens_per_gpu
        self.global_batch_size = global_batch_size
        self.pad_token_id = pad_token_id
        self.seed = seed
        self._epoch = 0

        # ensure the batch is even across all procs
        ds_lens = torch.zeros((dist.get_world_size(),), device=torch.device("cuda"))
        dist.all_reduce(ds_lens, op=dist.ReduceOp.SUM)
        ds_lens = ds_lens.tolist()
        if not all(l == ds_lens[0] for l in ds_lens):
            raise ValueError(f"Batch size is not even across all procs: {ds_lens}")

        # now we perform the initialization step
        rank = dist.get_rank()
        seq_lens = torch.zeros((dist.get_world_size(), len(dataset)), dtype=torch.long, device=torch.device("cuda"))
        seq_lens[rank] = torch.tensor(
            [item["seq_len"] for item in dataset], dtype=torch.long, device=torch.device("cuda")
        )
        dist.all_reduce(seq_lens, op=dist.ReduceOp.SUM)

        # now each node should have each other's sequence lengths
        self.all_seqs = seq_lens.tolist()

        # clear cache
        del seq_lens
        torch.cuda.empty_cache()

    def __iter__(self):
        """
        We couple the underlying sequence lengths together for the sake of being able to easily pack
        sequences across multiple ranks without having access to the data stored at each rank.
        """
        # assume that B x G % batch size and that batch_size % world_size
        local_batch_size = self.global_batch_size // dist.get_world_size()
        idxs = np.arange(len(self.dataset))
        seqs = np.array(self.all_seqs)

        # iterate through indices
        shuffled_idxs = np.random.RandomState(self.seed + self.epoch).permutation(idxs)
        for i in range(0, len(shuffled_idxs), local_batch_size):
            batch_idxs = shuffled_idxs[i : i + local_batch_size]
            batch_seqs = seqs[:, batch_idxs]  # pull out the indices for this batch at each rank

            # this is redundantly computed on each rank, but the local batch size is actually small so it's not a problem
            rank_minibatches = []
            for rank in range(dist.get_world_size()):
                # since we keep each rank independent (samples cannot move across ranks), num_ranks=1, rank=0
                minibatches = batch_lengths_to_minibatches_lpt(
                    batch_seqs[rank].tolist(), self.max_tokens_per_gpu, num_ranks=1, rank=0
                )
                rank_minibatches.append(minibatches)

            # now we apply padding to each rank's minibatches
            max_mb_len = max(len(mb) for mb in rank_minibatches)
            for mb in rank_minibatches:
                if len(mb) < max_mb_len:
                    mb.extend([[-1]] * (max_mb_len - len(mb)))  # witchcraft to add empty minibatches

            # sanity check
            assert all(len(mb) == max_mb_len for mb in rank_minibatches)

            # # Print rank_minibatches for debugging
            # for curr_r in range(dist.get_world_size()):
            #     if curr_r == dist.get_rank():
            #         print(f"rank {curr_r} rank_minibatches: {rank_minibatches[curr_r]}")

            num_microbatches = len(rank_minibatches[dist.get_rank()])
            flattened_indices = [idx for sublist in rank_minibatches[dist.get_rank()] for idx in (sublist + [-67])]
            num_samples = sum(len(sublist) for sublist in rank_minibatches[dist.get_rank()])

            # try:
            assert len(flattened_indices) > 0
            assert isinstance(flattened_indices[0], int)

            # accounts for the delimeter + padding tokens
            assert len(flattened_indices) == num_samples + num_microbatches, (
                f"{len(flattened_indices)} != {(local_batch_size + num_microbatches)=}"
            )
            # except AssertionError as e:
            #     print(f"rank {dist.get_rank()} flattened_indices: {flattened_indices}")
            #     print(f"rank {dist.get_rank()} local_batch_size: {local_batch_size}")
            #     print(f"rank {dist.get_rank()} num_microbatches: {num_microbatches}")
            #     print(f"rank {dist.get_rank()} rank_minibatches: {rank_minibatches[dist.get_rank()]}")
            #     print(f"AssertionError: {e}")
            # finally:
            #     pass

            # # Print batch structure similar to training loop
            # batch_structure = [
            #     str(seqs[dist.get_rank()][batch_idxs[idx]]) if idx != -1 else "#"
            #     for idx in flattened_indices
            #     if idx != -67
            # ]
            # for curr_r in range(dist.get_world_size()):
            #     if curr_r == dist.get_rank():
            #         print(f"rank {curr_r}: [" + " ".join(batch_structure) + "]")

            # now we yield the batch
            yield flattened_indices

    def __len__(self):
        return len(self.all_seqs)

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch


def oleg_collate_fn(batch: list[dict], pad_token_id: int, max_tokens_per_gpu: int):
    # LPT should have packed the batch such that we can collate the items by accumulating the number of tokens
    # until we exceed max_tokens_per_gpu in order to reset onto the next batch
    microbatches = []
    microbatch = []

    # first item should never be a delimiter
    assert not batch[0]["is_delimiter"]

    padding_microbatches = []

    # collates batches into how we packed them
    for i, item in enumerate(batch):
        is_last = i + 1 == len(batch)
        item: JsonlDatasetEntry
        if item["is_delimiter"]:
            if microbatch:
                microbatches.append(microbatch)
            microbatch = []
            continue

        # by now the delimiter should have appended the non-padding batch or this is the first one
        if item["is_padding"]:
            padding_microbatches.append([item])
            microbatch = []
            continue

        # add to the microbatch
        microbatch.append(item)
        if is_last and len(microbatch):
            microbatches.append(microbatch)
            microbatch = []

    # clear any remaining batches
    assert len(microbatch) == 0

    # now we transform into the final format
    final_microbatches = [{"microbatch": collate_fn(mb, pad_token_id), "padding": False} for mb in microbatches]

    # add the remaining padding items
    if padding_microbatches:
        final_microbatches += [
            {"microbatch": collate_fn(padding_mb, pad_token_id), "padding": True} for padding_mb in padding_microbatches
        ]

    return final_microbatches


def create_oleg_grpo_dataloader(
    dataset: list[Sample], pad_token_id: int, max_tokens_per_gpu: int, global_batch_size: int
):
    assert (
        len(dataset) * dist.get_world_size()
    ) % global_batch_size == 0  # we need to have an equal amount of samples on every rank

    local_batch_size = global_batch_size // dist.get_world_size()
    assert global_batch_size % local_batch_size == 0

    ds = JsonlDataset(dataset, pad_token_id)
    sampler = OlegDistributedSampler(ds, max_tokens_per_gpu, global_batch_size, pad_token_id)
    _oleg_collate_fn = functools.partial(
        oleg_collate_fn, pad_token_id=pad_token_id, max_tokens_per_gpu=max_tokens_per_gpu
    )
    loader = DataLoader(
        ds,
        batch_sampler=sampler,
        collate_fn=_oleg_collate_fn,
    )
    return loader


# so we need to be smart here,
# every rank will have an identical amount of samples, but since we're going to be doing FFD batch packing,
# with variable sequences, there is no guarantee that the actual amount of accumulation steps that need to be made
# in each model's minibatches will be identical

# since the datasets aren't synchronized, each rank will need to pack its own samples independently
# and then share the number of microbatches with each other so that they can pack additional dummy batches
# where needed

# we need a flow like this:
#
# rank
#  0                    ==>  D0                        ===> [[s1, s2], [s3]], ...                                [2, ...]
#  1   ==> load_dataset ==>  D1  ==> calculate packing ===>  [[s1, s2, s3]]   ... ==> calculate num forwards ==> [1, ...]
#  2                    ==>  D2                        ===> [[s1] [s2] [s3]], ...                                [3, ...]
# ...
# ---------------------------------------------------------------------------------------------
# rank
#  0  [2, ...]                                       [(1, 2)]
#  1  [1, ...] ==> batches become samples w/ IDs ==> [(1, 1)] ==> ranks share their lists, global packer decides sample coupling
#  2  [3, ...]                                       [(1, 3)]
# ---------------------------------------------------------------------------------------------
# rank
#  0  [(1, 2), ...]                         [..., (1, 2), ...]
#  1  [(1, 1), ...] samples are coupled ==> [(1, 1), ........] ==>
#  2  [(1, 3), ...]                         [........, (1, 3)]


class ProblemDataset(torch.utils.data.Dataset):
    def __init__(self, ds: datasets.Dataset):
        self.dataset = ds

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Problem:
        item = self.dataset[index]
        return Problem(
            answer=item["answer"],
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

    @classmethod
    def from_gsm8k(cls, eval_split: float = 0.0, seed: int = 67):
        ds = datasets.load_dataset("openai/gsm8k", name="main", split="train")

        # convert it into the problem format
        ds = ds.rename_columns({"question": "problem"})

        def _get_answers(sample):
            # gsm8k will produce the answers in the order needed to complete the problem,
            # so the final entry in the match will be the problem answer
            answers = re.findall("<<(.+)>>", sample["answer"])
            alt_matches = re.findall("#### (.+)", sample["answer"])
            if answers:
                answer = answers[-1].split("=")[-1]  # format: '12*2=24', '8/2=4', etc
            elif alt_matches:
                answer = alt_matches[-1]
            else:
                raise ValueError("failed to find matching samples")

            # this is the second format
            return {"answer": float(answer.replace(",", ""))}

        train_dataset = ds.map(_get_answers)
        eval_dataset = None

        if eval_split > 0:
            dataset_dict = train_dataset.train_test_split(test_size=eval_split, seed=seed)
            train_dataset = dataset_dict["train"]
            eval_dataset = dataset_dict["test"]

        return (ProblemDataset(train_dataset), ProblemDataset(eval_dataset) if eval_dataset else None)


def problem_collate_fn(batch: list[Problem]) -> list[Problem]:
    return batch
