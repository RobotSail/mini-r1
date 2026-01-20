import torch
import datasets
import random

# from distributed_packer import pack_for_distributed_training
from type_defs import Problem, Sample, JsonlDatasetEntry
import re
import torch.distributed as dist
from torch.utils.data import Sampler, DataLoader, BatchSampler
import numpy as np
import functools


def random_problems(
    seed: int = 42, num_problems: int = 20, min_num: int = 1, max_num: int = 100
) -> list[Problem]:
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
    problems = random_problems(
        seed=seed, num_problems=num_problems, min_num=min_num, max_num=max_num
    )

    # Convert list of Problem objects to dataset
    problems_dict = [problem.model_dump() for problem in problems]
    dataset = datasets.Dataset.from_list(problems_dict)
    dataset = dataset.map(
        lambda x: {
            "messages": (
                [{"role": "system", "content": system_msg}] if system_msg else []
            )
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
            "logprobs": [0.0],
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

        # Diagnostic: track advantage statistics during dataset creation
        all_advantages = []

        # we must flatten the batch
        for sample in dataset:
            # these are the prompt input IDs
            prompt_input_ids = sample.input_ids
            for rollout in sample.rollouts:
                # skip
                if not rollout.logprobs:
                    raise ValueError(
                        "found a rollout without logprobs, this shouldn't happen"
                    )
                    continue

                # DIAGNOSTIC: Verify token_ids match logprob tokens
                # logprob_token_ids = [lp.token for lp in rollout.logprobs]
                # if len(rollout.token_ids) != len(logprob_token_ids):
                #     print(f"[DATASET DEBUG] LENGTH MISMATCH: token_ids={len(rollout.token_ids)} vs logprob_tokens={len(logprob_token_ids)}")
                # else:
                #     mismatches = [(i, tid, lptid) for i, (tid, lptid) in enumerate(zip(rollout.token_ids, logprob_token_ids)) if tid != lptid]
                #     if mismatches:
                #         print(f"[DATASET DEBUG] TOKEN ID MISMATCH at {len(mismatches)}/{len(rollout.token_ids)} positions:")
                #         for i, tid, lptid in mismatches[:10]:  # Show first 10
                #             print(f"  Position {i}: token_ids[{i}]={tid} != logprob.token={lptid}")
                #     else:
                #         print(f"[DATASET DEBUG] All {len(rollout.token_ids)} token IDs match between token_ids and logprobs ✓")

                # this should be good enough
                input_ids = prompt_input_ids[:] + rollout.token_ids[:]
                grpo_mask = [False] * len(prompt_input_ids) + [True] * len(
                    rollout.token_ids[:]
                )
                logprobs_seq = [0.0] * len(prompt_input_ids) + [
                    lp.logprob for lp in rollout.logprobs
                ]

                # now we shift so offset is correct and we predict causal
                # Suppose our input was "The Cat" and model generated "Sat Down EOS"
                input_ids = input_ids[
                    :-1
                ]  #      ['The', 'Cat', 'Sat', 'Down', 'EOS']  --> ['The', 'Cat', 'Sat', 'Down']

                # Creates a sequence of           [PAD,   'Sat', 'Down', 'EOS']
                logprob_tokens: list[int] = [pad_token_id] * len(
                    prompt_input_ids[1:]
                ) + [lp.token for lp in rollout.logprobs]
                grpo_mask = grpo_mask[
                    1:
                ]  # [False, False, True, True, True]       --> [False, True, True, True]
                logprobs_seq = logprobs_seq[
                    1:
                ]  # [1.0,   1.0,   0.35,  0.93,  0.23]   --> [1.0, 0.35, 0.93, 0.23]
                num_logprobs = len(rollout.logprobs)  # produces 3

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


# @dataclass
# class DataCollatorWithFlattening(DefaultDataCollator):
#     """
#     Data collator used for padding free approach. Does the following:

#     - concatenates the entire mini batch into single long sequence of shape [1, total_tokens]
#     - uses `separator_id` to separate sequences within the concatenated `labels`, default value is -100
#     - no padding will be added, returns `input_ids`, `labels` and `position_ids` by default
#     - optionally returns the kwargs contained in FlashAttentionKwargs
#     - optionally returns seq_idx indicating which sequence each token belongs to

#     <Tip warning={true}>

#     Using `DataCollatorWithFlattening` will flatten the entire mini batch into single long sequence.
#     Make sure your attention computation is able to handle it!

#     </Tip>
#     """

#     def __init__(
#         self,
#         *args,
#         return_position_ids=True,
#         separator_id=-100,
#         return_flash_attn_kwargs=False,
#         return_seq_idx=False,
#         **kwargs,
#     ):
#         super().__init__(*args, **kwargs)
#         self.return_position_ids = return_position_ids
#         self.separator_id = separator_id
#         self.return_flash_attn_kwargs = return_flash_attn_kwargs
#         self.return_seq_idx = return_seq_idx
#         self._int_64_keys = {"labels", "position_ids", "input_ids"}
#         self._batch_dim_keys = {"labels", "position_ids", "input_ids", "seq_idx"}
#         self._py_int_keys = {"max_length_q", "max_length_k"}

#     def __call__(self, features, return_tensors=None, separator_id=None):
#         if return_tensors is None:
#             return_tensors = self.return_tensors
#         if separator_id is None:
#             separator_id = self.separator_id
#         is_labels_provided = "labels" in features[0]
#         batch = {"input_ids": [], "labels": []}
#         if self.return_position_ids:
#             batch.update({"position_ids": []})
#         if self.return_seq_idx:
#             batch.update({"seq_idx": []})
#         if self.return_flash_attn_kwargs:
#             cu_seq_lens = [0]
#             max_length = 0
#         for seq_idx, sample in enumerate(features):
#             input_ids = sample["input_ids"]
#             batch["input_ids"] += input_ids
#             if is_labels_provided:
#                 batch["labels"] += [separator_id] + sample["labels"][1:]
#             else:
#                 batch["labels"] += [separator_id] + input_ids[1:]
#             if self.return_position_ids:
#                 batch["position_ids"] += list(range(len(input_ids)))
#             if self.return_seq_idx:
#                 batch["seq_idx"] += [seq_idx for _ in range(len(input_ids))]
#             if self.return_flash_attn_kwargs:
#                 cu_seq_lens.append(cu_seq_lens[-1] + len(input_ids))
#                 max_length = max(max_length, len(input_ids))

#         if self.return_flash_attn_kwargs:
#             batch["cu_seq_lens_q"] = batch["cu_seq_lens_k"] = cu_seq_lens
#             batch["max_length_q"] = batch["max_length_k"] = max_length

#         # FlashAttentionKwargs and seq_idx are expected to be int32s.
#         if return_tensors == "pt":
#             import torch

#             data_cls = torch.tensor
#             dtype_64 = torch.int64
#             dtype_32 = torch.int32
#         elif return_tensors == "np":
#             data_cls = np.array
#             dtype_64 = np.int64
#             dtype_32 = np.int32
#         else:
#             raise ValueError(f'return_tensors must be one of ("pt", "np"), {return_tensors=} not supported')

#         for k, v in batch.items():
#             if k in self._batch_dim_keys:
#                 v = [v]
#             # Flash attention max_len_{q,k} are python ints
#             if k not in self._py_int_keys:
#                 batch[k] = data_cls(v, dtype=dtype_64 if k in self._int_64_keys else dtype_32)

#         return batch


def collate_fn(batch: list[dict], pad_token_id: int):
    """
    Collate with proper attention masking to prevent cross-sequence attention.

    Creates a block-diagonal causal attention mask so each packed sequence
    can only attend to itself, matching vLLM's isolated generation context.
    """
    num_tokens_in_batch = 0
    advantages = []
    input_ids = []
    logprobs = []
    grpo_mask = []
    position_ids: list[torch.Tensor] = []
    logprob_ids: list[torch.Tensor] = []

    # Track sequence boundaries for attention mask construction
    sequence_lengths = []

    # in order to average over the sequence length properly
    # we calculate the total batch size here
    grpo_scalars = []

    # okay so we need to do it a little differently.
    # for flash-attention 2 and padding-free training, we instead provide
    # - input_ids (as-is)
    # - position IDs (range up to N) for each sequence always reseting at 0 for the next seq
    # - seq idx - list of the length of input IDs set to the index of the sequence we are processing
    # - cu_seq_lens - a list of the cumulative sequence lengths at each index, should be like: [0, N1, N1 + N2, ...]
    # - cu_seq_lens_k, cu_seq_lens_q : the same list (cu_seq_lens)
    # - max_length_k, max_length_q : the max length in the index

    max_length = 0
    cu_seq_lens = [0]
    seq_idxs = []

    for i, item in enumerate(batch):
        last_item = i + 1 == len(batch)
        seq_len = item["input_ids"].numel()

        max_length = max(max_length, seq_len)

        input_ids += [item["input_ids"]]
        logprob_ids += [item["logprob_ids"]]
        logprobs += [item["logprobs"]]
        grpo_mask += [item["grpo_mask"]]
        position_ids += [torch.tensor(range(seq_len))]
        sequence_lengths.append(seq_len)
        cu_seq_lens.append(cu_seq_lens[-1] + seq_len)

        seq_idxs += [i] * seq_len

        # we will use these to divide the final logprobs sequence
        scalars = torch.masked_fill(
            torch.ones_like(item["input_ids"]),
            item["grpo_mask"],
            value=item["num_logprobs"],
        )
        advantages += [
            torch.full_like(item["grpo_mask"], item["advantage"], dtype=torch.float32)
        ]
        grpo_scalars += [scalars]

    # Create block-diagonal causal attention mask
    # Each sequence gets a causal mask within its block, zeros elsewhere
    total_length = sum(sequence_lengths)

    # For HuggingFace models, attention_mask: 1 = attend, 0 = mask out
    # We'll create a 2D mask that gets broadcast during attention
    attention_mask = torch.zeros((total_length, total_length), dtype=torch.bool)

    start_idx = 0
    for seq_len in sequence_lengths:
        end_idx = start_idx + seq_len
        # Fill in causal mask for this sequence block
        for i in range(start_idx, end_idx):
            # Token i can attend to tokens [start_idx, i] (causal)
            attention_mask[i, start_idx : i + 1] = True
        start_idx = end_idx

    final_item = {
        "input_ids": torch.cat(input_ids).detach(),
        "position_ids": torch.cat(position_ids).detach(),
        "attention_mask": attention_mask.detach(),
        "advantages": torch.cat(advantages).detach(),
        "logprobs": torch.cat(logprobs).detach(),
        "logprob_ids": torch.cat(logprob_ids).detach(),
        "grpo_mask": torch.cat(grpo_mask).detach(),
        "scalars": torch.cat(grpo_scalars).detach(),
        "cu_seq_lens": torch.tensor(cu_seq_lens).detach(),
        "seq_idx": torch.tensor([seq_idxs]).detach(),
        "max_length_q": int(max_length),
        "max_length_k": int(max_length),
        "cu_seq_lens_q": torch.tensor(cu_seq_lens, dtype=torch.int32).detach(),
        "cu_seq_lens_k": torch.tensor(cu_seq_lens, dtype=torch.int32).detach(),
    }
    return final_item


class OlegDistributedSampler(BatchSampler):
    def __init__(
        self,
        dataset: JsonlDataset,
        max_tokens_per_gpu: int,
        global_batch_size: int,
        pad_token_id: int,
        seed: int = 67,
    ):
        self.dataset = dataset
        self.max_tokens_per_gpu = max_tokens_per_gpu
        self.global_batch_size = global_batch_size
        self.pad_token_id = pad_token_id
        self.seed = seed
        self._epoch = 0
        self._rank = dist.get_rank()  # global rank for data parallel is correct

        # ensure the batch is even across all procs
        ds_lens = torch.zeros((dist.get_world_size(),), device=torch.device("cuda"))
        dist.all_reduce(ds_lens, op=dist.ReduceOp.SUM)
        ds_lens = ds_lens.tolist()
        if not all(l == ds_lens[0] for l in ds_lens):
            raise ValueError(f"Batch size is not even across all procs: {ds_lens}")

        # now we perform the initialization step
        rank = dist.get_rank()
        seq_lens = torch.zeros(
            (dist.get_world_size(), len(dataset)),
            dtype=torch.long,
            device=torch.device("cuda"),
        )
        seq_lens[rank] = torch.tensor(
            [item["seq_len"] for item in dataset],
            dtype=torch.long,
            device=torch.device("cuda"),
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
            batch_seqs = seqs[
                :, batch_idxs
            ]  # pull out the indices for this batch at each rank

            # SIMPLIFIED: Use random packing instead of LPT to avoid same-group clustering
            rank_minibatches = []
            for rank_idx in range(dist.get_world_size()):
                minibatches = []
                current_batch = []
                current_tokens = 0

                # the indices should be identical on all ranks but the lengths vary rank-to-rank
                for idx, length in zip(
                    batch_idxs.tolist(), batch_seqs[rank_idx].tolist()
                ):
                    if (
                        current_tokens + length > self.max_tokens_per_gpu
                        and current_batch
                    ):
                        # Batch is full, start a new one
                        minibatches.append(current_batch)
                        current_batch = [idx]
                        current_tokens = length
                    else:
                        current_batch.append(idx)
                        current_tokens += length

                # Add remaining batch
                if current_batch:
                    minibatches.append(current_batch)

                rank_minibatches.append(minibatches)

            # now we apply padding to each rank's minibatches
            max_mb_len = max(len(mb) for mb in rank_minibatches)
            for mb in rank_minibatches:
                if len(mb) < max_mb_len:
                    mb.extend(
                        [[-1]] * (max_mb_len - len(mb))
                    )  # witchcraft to add empty minibatches

            # sanity check
            assert all(len(mb) == max_mb_len for mb in rank_minibatches)

            # # Print rank_minibatches for debugging
            # for curr_r in range(dist.get_world_size()):
            #     if curr_r == dist.get_rank():
            #         print(f"rank {curr_r} rank_minibatches: {rank_minibatches[curr_r]}")

            num_microbatches = len(rank_minibatches[dist.get_rank()])
            flattened_indices = [
                idx
                for sublist in rank_minibatches[dist.get_rank()]
                for idx in (sublist + [-67])
            ]
            num_samples = sum(
                len(sublist) for sublist in rank_minibatches[dist.get_rank()]
            )

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
        # keep this
        all_lens = [len(self.all_seqs[r]) for r in range(dist.get_world_size())]
        assert [all_lens[0] == l for l in all_lens], (
            f"expected even amount of sequence lengths across all ranks but found disparity: {all_lens=}"
        )

        # this should be the accurate calculation though
        local_batch_size = self.global_batch_size // dist.get_world_size()

        # len(dataset) is the length of the local sharded dataset
        return len(self.dataset) // local_batch_size

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
    padding_microbatch = []

    # collates batches into how we packed them
    for i, item in enumerate(batch):
        is_last = i + 1 == len(batch)
        item: JsonlDatasetEntry

        # by now the delimiter should have appended the non-padding batch or this is the first one
        if item["is_padding"]:
            padding_microbatch.append(item)
        elif not item["is_delimiter"]:
            # add to the microbatch
            microbatch.append(item)

        if is_last or item["is_delimiter"]:
            if microbatch:
                microbatches.append(microbatch)
                microbatch = []
            elif padding_microbatch:
                padding_microbatches.append(padding_microbatch)
                padding_microbatch = []
            else:
                raise ValueError(
                    "received an empty delimeter without the microbatch or padding being populated"
                )

    # clear any remaining batches
    assert len(microbatch) == 0

    # number of trainable items in local minibatch
    items_in_local_minibatch = sum(len(mb) for mb in microbatches)

    # now we transform into the final format
    final_microbatches = [
        {"microbatch": collate_fn(mb, pad_token_id), "padding": False}
        for mb in microbatches
    ]

    # add the remaining padding items
    if padding_microbatches:
        final_microbatches += [
            {"microbatch": collate_fn(padding_mb, pad_token_id), "padding": True}
            for padding_mb in padding_microbatches
        ]

    return {
        "microbatches": final_microbatches,
        "num_samples_in_local_minibatch": items_in_local_minibatch,
    }


def create_oleg_grpo_dataloader(
    dataset: list[Sample],
    pad_token_id: int,
    max_tokens_per_gpu: int,
    global_batch_size: int,
):
    # Flatten samples to rollouts first
    # ds = JsonlDataset(dataset, pad_token_id)

    # Check that total rollouts can be evenly divided by global_batch_size
    # (Note: len(ds) is rollouts per rank, not total problems)
    # total_rollouts = len(ds) * dist.get_world_size()
    # assert total_rollouts % global_batch_size == 0, (
    #     f"Total rollouts ({total_rollouts}) must be divisible by global_batch_size ({global_batch_size})"
    # )

    local_batch_size = global_batch_size // dist.get_world_size()
    assert global_batch_size % local_batch_size == 0

    ds = JsonlDataset(dataset, pad_token_id)
    sampler = OlegDistributedSampler(
        ds, max_tokens_per_gpu, global_batch_size, pad_token_id
    )
    _oleg_collate_fn = functools.partial(
        oleg_collate_fn,
        pad_token_id=pad_token_id,
        max_tokens_per_gpu=max_tokens_per_gpu,
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
            dataset_dict = train_dataset.train_test_split(
                test_size=eval_split, seed=seed
            )
            train_dataset = dataset_dict["train"]
            eval_dataset = dataset_dict["test"]

        return (
            ProblemDataset(train_dataset),
            ProblemDataset(eval_dataset) if eval_dataset else None,
        )

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
            dataset_dict = train_dataset.train_test_split(
                test_size=eval_split, seed=seed
            )
            train_dataset = dataset_dict["train"]
            eval_dataset = dataset_dict["test"]

        return (
            ProblemDataset(train_dataset),
            ProblemDataset(eval_dataset) if eval_dataset else None,
        )


def problem_collate_fn(batch: list[Problem]) -> list[Problem]:
    return batch
