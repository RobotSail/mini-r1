from __future__ import annotations

import copy
import functools
from datetime import timedelta
from typing import TYPE_CHECKING
from torch._tensor import Tensor
import typing as t

if TYPE_CHECKING:
    from src.rewards import RewardFn
import openai
from torch.optim import AdamW
import torch.distributed as dist
import pydantic
import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from transformers import PreTrainedModel, PreTrainedTokenizerBase
import os
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
    FullyShardedDataParallel,
    CPUOffloadPolicy,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2Tokenizer,
    PreTrainedModel,
)
import typer
from datasets import Dataset
from huggingface_hub import split_torch_state_dict_into_shards
from transformers import AutoTokenizer
from safetensors.torch import save_file
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
)
import json
import httpx
import time
import asyncio
from openai import AsyncOpenAI
import openai.types.chat

from openai.types.chat.chat_completion import ChatCompletion


from utils import log_rank_0


def disable_dropout_in_model(model: torch.nn.Module) -> None:
    """Disable dropout in a model by setting dropout probability to 0."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        # Also handle attention dropout in transformers
        if hasattr(module, "attention_dropout"):
            module.attention_dropout = 0.0
        if hasattr(module, "hidden_dropout"):
            module.hidden_dropout = 0.0
        if hasattr(module, "resid_dropout"):
            module.resid_dropout = 0.0


class VllmNotReady(BaseException):
    def __init__(self, msg: str):
        self.message = msg
        super().__init__(self.message)


class Problem(pydantic.BaseModel):
    problem: str
    answer: int | float


class SamplingParams(pydantic.BaseModel):
    temperature: float
    top_p: float = 1.0
    top_k: float = 0.0
    repetition_penalty: float = 1.0
    max_new_tokens: int = 0


class Message(pydantic.BaseModel):
    role: str
    content: str


class TokenSample(pydantic.BaseModel):
    token: int
    logprob: float


class RolloutScore(pydantic.BaseModel):
    is_parsable: bool = False
    is_correct: bool = False
    reward: float = 0.0


class JsonlDatasetEntry(t.TypedDict):
    input_ids: list[int]
    logprobs: list[float]
    grpo_mask: list[bool]
    num_logprobs: int
    advantage: float
    logprob_ids: list[int]
    is_delimiter: bool
    is_padding: bool


class RolloutResult(pydantic.BaseModel):
    response: str
    token_ids: list[int]
    logprobs: list[TokenSample] | None = None
    score: RolloutScore | None = None
    advantage: float = 0.0

    @classmethod
    def from_problem_completion(
        cls,
        problem: Problem,
        choice: openai.types.chat.chat_completion.Choice,
        reward_fn: "RewardFn | None" = None,
    ):
        from src.rewards import DEFAULT_REWARD_FN

        reward_fn = reward_fn or DEFAULT_REWARD_FN

        logprobs = []
        if choice.logprobs is not None:
            logprobs = [
                TokenSample(token=lp.token.split(":")[-1], logprob=lp.logprob)
                for lp in choice.logprobs.content
            ]

        return RolloutResult(
            response=choice.message.content,
            token_ids=choice.token_ids,
            logprobs=logprobs,
            score=reward_fn(choice.message.content, problem.answer),
        )


class Sample(pydantic.BaseModel):
    problem: Problem
    # system_prompt: str | None = None
    rollouts: list[RolloutResult] = pydantic.Field(default_factory=list)
    input_ids: list[int]

    @staticmethod
    def calculate_advantage(group: list[RolloutResult], dr_grpo: bool = False):
        """
        Given the group of rollouts, calculate the group advantage of each
        sample and populate it in-place.

        GRPO equation: A_i = (r_i - avg(r)) / (std(r))
        """
        eps = 1e-8
        avg = sum(r.score.reward for r in group) / len(group)
        var = sum((r.score.reward - avg) ** 2 for r in group) / len(group)
        std = var**0.5

        # if std < eps (because all rewards are equal) we use the std trick
        # of setting group advantage to 0
        enable_std_trick = std < eps

        # GRPO simple advantage
        for rollout in group:
            # dr. grpo eliminates standard deviation
            if dr_grpo:
                rollout.advantage = rollout.score.reward - avg
                continue

            # regular grpo
            if enable_std_trick:
                rollout.advantage = 0.0
            else:
                adv = (rollout.score.reward - avg) / (std + eps)
                # CRITICAL: Clamp advantages to prevent extreme policy updates
                rollout.advantage = max(-10.0, min(10.0, adv))

        nonzero_adv = sum(1 for r in group if abs(r.advantage) > 1e-6)
        rewards_str = ", ".join(
            f"{r.score.reward:.1f}" for r in group[:5]
        )  # Show first 5
        if len(group) > 5:
            rewards_str += "..."
        log_rank_0(
            f"Group: {nonzero_adv}/{len(group)} non-zero adv | rewards=[{rewards_str}] | std={std:.4f} | trigger={enable_std_trick}"
        )

    @classmethod
    def from_chat_completion(
        cls, problem: Problem, completion: ChatCompletion
    ) -> "Sample":
        """
        Makes a call to the current policy to generate a sample from the given
        problem.
        """
        input_ids = completion.prompt_token_ids
        responses: list[RolloutResult] = []

        # calculate advantage of group and set it in-place
        for choice in completion.choices:
            responses.append(RolloutResult.from_problem_completion(problem, choice))
        cls.calculate_advantage(responses)
        return cls(input_ids=input_ids, rollouts=responses, problem=problem)

    # def calculate_scores(self):
    #     # GRPO equation:
    #     # A_i = (r_i - avg(r)) / std(r)
    #     # Dr.GRPO Equation:
    #     # A_i = r_i - avg(r)
    #     for rollout in self.rollouts:
    #         # Defaults; if we cannot parse then it is not correct. If we can parse, it is not necessarily correct.
    #         rollout.is_parsable = False
    #         rollout.is_correct = False

    #         # reset it here just for good measure
    #         rollout.reward = 0

    #         # check if the response has any answers at all
    #         matches = answer_pattern.findall(rollout.response)

    #         # we only want 1 of these
    #         parsed_nums: list[int | float] = []  #
    #         for match in matches:  # this is already bad
    #             try:
    #                 answer = parse_number(match)
    #             except Exception as e:
    #                 print(f"failed to parse text from answer tags: {e}")
    #             else:
    #                 parsed_nums.append(answer)

    #         if len(parsed_nums) == 1:
    #             answer = parsed_nums[0]
    #             rollout.is_parsable = True
    #             rollout.is_correct = group.problem.answer == answer
    #         if len(parsed_nums) > 1:
    #             rollout.reward -= 0.1
    #         elif rollout.is_parsable:
    #             # okay, we WANT the model to produce more answers like this
    #             # but we don't want to overweight this or give sparse rewards
    #             # so we will assign a reward here of +1
    #             rollout.reward += 0.1

    #         if rollout.is_correct:
    #             rollout.reward += 1  # huge reward for getting it right


class Hyperparameters(pydantic.BaseModel):
    """
    Hyperparameters for GRPO, defaults are set to small/debug values values
    """

    lr: float
    msl_post: int
    msl_pre: int
    msl_jump_at_step: int
    max_steps: int
    batch_size: int = 1
    group_size: int = 8
    inner_epochs: int = 1
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    adamw_wd: float = 0.01

    # packing length for training
    max_tokens_per_gpu: int

    # msl_pre: int =

    update_ref_policy_every_n_steps: int = -1  # off by default
    dr_grpo: bool = False

    # change this one to a ratio
    inner_batch_size: int
    eps: float = 0.1  # or 0.2 as in deepseek's original paper
    kl_penalty_strength: float = 0.01  # start with a lower value


def requires_vllm_ready(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.wait_for_vllm_to_be_ready()
        return func(self, *args, **kwargs)

    return wrapper


class TrainingContext(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    model_name: str
    system_msg: str | None = None
    output_dir: str | None = None
    optimizer: torch.optim.Optimizer | None = None
    model: PreTrainedModel | None = None
    ref_model: PreTrainedModel | None = None
    tokenizer: PreTrainedTokenizerBase | None = None
    hparams: Hyperparameters
    device: torch.device = pydantic.Field(
        default_factory=lambda: torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    )
    sampling_params: SamplingParams
    device_mesh: DeviceMesh | None = None  # this will start out being none
    ref_model_cpu_offload: bool = False  # CPU offload for ref model to save GPU memory
    world_size: int
    vllm_url: str
    vllm_model_name: str
    vllm_model_dir: str
    dataset: str
    train_path: str | None = None
    eval_split: float = 0.0
    output_dir: str | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vllm_is_ready = False
        self._policy_has_changed = False

    def valid_save_dir(self) -> bool:
        if not self.output_dir:
            return False

        # Try to create the directory if it doesn't exist
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            return True
        except (OSError, PermissionError):
            return False

    def load_models(self):
        # device setup

        # initialize model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            attn_implementation="flash_attention_2",
            dtype=torch.float32,
        )

        # this is a frozen model which we do not update
        # Load fresh copy - use force_download or different approach to avoid any sharing
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            attn_implementation="flash_attention_2",
            dtype=torch.float32,
        )

        disable_dropout_in_model(self.model)
        disable_dropout_in_model(self.ref_model)

        self.ref_model.eval()
        self.ref_model.requires_grad_(False)

        # Debug: Verify models have independent parameters before FSDP wrapping
        self.tokenizer: Qwen2Tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # align tokenizer and tokens
        for m in [self.model, self.ref_model]:
            if self.tokenizer.pad_token_id and not m.config.pad_token_id:
                m.config.pad_token_id = self.tokenizer.pad_token_id
                typer.secho(
                    f"model '{self.model_name}' doesn't have a pad_token_id, setting it to {self.tokenizer.pad_token_id}",
                    fg=typer.colors.BRIGHT_BLUE,
                )

    def create_device_mesh(self):
        self.device_mesh = init_device_mesh(
            "cuda", mesh_shape=(self.world_size,), mesh_dim_names=("fsdp",)
        )

    def wrap_models_with_fsdp2(self):
        mp_training_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,  # training model needs more range due to activations
            reduce_dtype=torch.float32,
        )

        # Disable HuggingFace cache if present
        if hasattr(self.model, "config"):
            try:
                self.model.config.use_cache = False
            except Exception as e:
                print(
                    f"WARNING: Failed to disable HuggingFace cache for model {self.model.__class__.__name__}: {e}"
                )

        # select layers
        layers = self.model.model.layers

        # apply activation checkpointing
        for i, layer in enumerate(layers):
            layers[i] = ptd_checkpoint_wrapper(layer, preserve_rng=True)

        # now wrap each module with fsdp2
        for idx, block in enumerate(layers):
            reshard = idx < len(layers) - 1
            fully_shard(
                block,
                mesh=self.device_mesh,
                mp_policy=mp_training_policy,
                reshard_after_forward=reshard,
            )
        fully_shard(
            self.model,
            mesh=self.device_mesh,
            mp_policy=mp_training_policy,
            reshard_after_forward=False,
        )

        # Wrap ref_model with FSDP for memory efficiency
        # Note: Optimizer must be created AFTER this to avoid stale parameter references
        layers = self.ref_model.model.layers
        ref_mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        # Optional CPU offload for ref model - keeps params on CPU, loads to GPU during forward
        ref_offload_policy = (
            CPUOffloadPolicy(pin_memory=True) if self.ref_model_cpu_offload else None
        )
        if self.ref_model_cpu_offload:
            log_rank_0("Using CPU offload for reference model")

        for idx, block in enumerate(layers):
            fully_shard(
                block,
                mesh=self.device_mesh,
                mp_policy=ref_mp_policy,
                offload_policy=ref_offload_policy,
                reshard_after_forward=True,
            )
        fully_shard(
            self.ref_model,
            mesh=self.device_mesh,
            mp_policy=ref_mp_policy,
            offload_policy=ref_offload_policy,
            reshard_after_forward=True,
        )

        # Debug: Verify models have independent parameters after FSDP wrapping
        model_param = next(self.model.parameters())
        ref_param = next(self.ref_model.parameters())
        # Check if they're the same tensor object
        same_object = model_param is ref_param
        # Check if values are identical (they should be at init, but diverge after training)
        values_equal = torch.equal(model_param, ref_param)
        log_rank_0(
            f"DEBUG: After FSDP - same_object: {same_object}, values_equal: {values_equal}"
        )

        # Create optimizer AFTER FSDP wrapping - this is critical!
        # See https://github.com/pytorch/pytorch/issues/149205
        self.optimizer = AdamW(
            params=self.model.parameters(),
            lr=self.hparams.lr,
            betas=self.hparams.adamw_betas,
            weight_decay=self.hparams.adamw_wd,
        )
        log_rank_0("Optimizer created after FSDP wrapping")

    @requires_vllm_ready
    def update_vllm_policy(self):
        """
        Update the policy model served at vLLM with the current policy
        """
        if not self._policy_has_changed:
            log_rank_0("skipping vllm update, policy hasn't changed")
            return

        # first we have to write the checkpoint to the directory where we expect
        self.save_model(self.vllm_model_dir, self.model, self.tokenizer, torch.float16)
        torch.cuda.empty_cache()

        # Next, we need to force vLLM to reload the weights
        if dist.get_rank() == 0:
            log_rank_0("reloading inference server weights")
            resp = httpx.post(
                f"{self.vllm_url}/collective_rpc", json={"method": "reload_weights"}
            )
            log_rank_0("waiting for inference server to reload weights")
            resp.raise_for_status()  # fail fast if something went wrong

            # clear stale prefix cache
            log_rank_0("clearing kv cache from inference server")
            resp = httpx.post(f"{self.vllm_url}/reset_prefix_cache")
            log_rank_0("waiting for inference server to clear kv cache")
            resp.raise_for_status()
        dist.barrier()
        log_rank_0("successfully reloaded policy")

        # fresh update
        self._policy_has_changed = False

    @staticmethod
    def save_model(
        output_dir: str, fsdp_model, tokenizer, save_dtype: torch.dtype = None
    ):
        """
        Save the given FSDP Model as a checkpoint in HF Format.

        Args:
            fsdp_model (str): The model to save.
            samples_seen (int): The number of samples seen so far.
            output_dir (str): The directory to save the model.
            model_name_or_path (str): The model name or path.
            suffix (str | None): Optional suffix to add to the checkpoint directory name.
        """
        global_rank = torch.distributed.get_rank()

        # Add suffix to directory name if provided
        os.makedirs(output_dir, exist_ok=True)

        # NOTE(osilkin):
        # Here, we gather the model's state-dict and offload it onto the CPU
        # The downside with this approach is that it requires recomputing
        # each OSFT parameter on the CPU.
        # This can be optimized by modifying the `prepare_state_dict_for_save` function so that it
        # processes weights on the GPU device in batches before de-allocating the memory being consumed
        # Users may also face issues here if they lack the CPU memory required to store the original
        # FP32 state dict on CPU.

        state_dict = get_model_state_dict(
            fsdp_model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
                broadcast_from_rank0=False,
            ),
        )
        inner = getattr(fsdp_model, "module", fsdp_model)
        # save in whatever data type is stored on the model config
        # by now the `torch_dtype` attribute has been set to some value
        save_dtype = (
            next(p.dtype for p in fsdp_model.parameters())
            if save_dtype is None
            else save_dtype
        )

        # NOTE(osilkin): This save function could be further optimized for quicker checkpoints:
        #
        # FSDP2 provides a distributed checkpoint API, which allows all shards to
        # save their respective format, which can be post-processed afterwards
        # to recover the model and optionally the optimizer states.
        #
        # However; switching to this format would require:
        # 1.) Converting checkpoints into HF format after training completes
        # 2.) All nodes having access to the same write location, which is also synchronized for us
        #     to actually export the checkpoints properly.

        torch.distributed.barrier()
        if global_rank == 0:
            # Model format conversion (GPT-OSS vs standard)
            # Once we have all of our parameters, we need to ensure they're stored in BF16
            # so checkpoints aren't terrible heavy. We have to do this _after_ `prepare_state_dict_for_save`
            # has been called so we don't lose fidelity.
            notified_about_dtype = False
            cpu_device = torch.device("cpu")
            # Standard conversion to bf16 and CPU
            for k, v in state_dict.items():
                if v.dtype != save_dtype:
                    if not notified_about_dtype:
                        log_rank_0(
                            f"⚠️  Warning: Found tensor {k} with dtype {v.dtype}, casting to {save_dtype}"
                        )
                        notified_about_dtype = True
                    state_dict[k] = v.to(dtype=save_dtype, device=cpu_device)

            # All saving operations
            pattern = "model{suffix}.safetensors"
            index_name = "model.safetensors.index.json"

            # Shard splitting
            split = split_torch_state_dict_into_shards(
                state_dict,
                filename_pattern=pattern,
                max_shard_size="5GB",
            )
            # Save shards
            for filename, tensors in split.filename_to_tensors.items():
                shard = {k: state_dict[k] for k in tensors}
                path = os.path.join(output_dir, filename)
                save_file(shard, path)

            # Save index if sharded
            if split.is_sharded:
                index = {
                    "metadata": split.metadata,
                    "weight_map": split.tensor_to_filename,
                }
                with open(os.path.join(output_dir, index_name), "w") as f:
                    json.dump(index, f, indent=2, sort_keys=True)
            # Standard config save for non-GPT-OSS models
            inner.config.to_json_file(os.path.join(output_dir, "config.json"))
            tokenizer.save_pretrained(output_dir)

        torch.distributed.barrier()

    @property
    def vllm_is_ready(self) -> bool:
        if self._vllm_is_ready:
            return True

        server_is_ready = False
        unexpected_error = False
        err = None
        try:
            if dist.get_rank() == 0:
                # first we need to make sure sure vLLM itself is running
                response = httpx.get(f"{self.vllm_url}/health", timeout=1)
                if response.status_code != 200:
                    raise VllmNotReady(
                        f"health check return non-200 code: {response.status_code}"
                    )

                # now check if our model is there
                response = httpx.get(f"{self.vllm_url}/v1/models", timeout=1)
                if response.status_code != 200:
                    raise VllmNotReady(
                        f"model check return non-200 code: {response.status_code}"
                    )

                models = response.json().get("data", [])
                model_names = [model.get("id") for model in models]

                if self.vllm_model_name not in model_names:
                    raise VllmNotReady(
                        f"Model '{self.vllm_model_name}' not found in vLLM. Available models: {model_names}",
                    )
                server_is_ready = True

        except (httpx.RequestError, httpx.TimeoutException, VllmNotReady) as e:
            log_rank_0(f"vLLM health check failed: {e}")
        except Exception as e:
            log_rank_0(f"hit unexpected error: {e}")
            unexpected_error = True
        finally:
            # this is where everyone waits
            dist.barrier()

        # now we can communicate findings
        log_rank_0(f"communicating vllm status to group")
        readiness_tuple = torch.tensor(
            [server_is_ready, unexpected_error], dtype=torch.bool, device=self.device
        )
        dist.broadcast(readiness_tuple, src=0)

        server_is_ready, unexpected_error = (
            readiness_tuple[0].item(),
            readiness_tuple[1].item(),
        )
        if unexpected_error:
            raise RuntimeError(f"experienced unexpected error during training")

        self._vllm_is_ready = server_is_ready
        return server_is_ready

    def wait_for_vllm_to_be_ready(self):
        while not self.vllm_is_ready:
            log_rank_0("vllm is not ready yet, checking again in 3s")
            time.sleep(3)

    async def _generate_samples_for_problem(
        self,
        client: AsyncOpenAI,
        problem: Problem,
        n_samples: int,
        include_logprobs: bool,
        max_seq_len: int,
    ) -> Sample:
        """
        Makes a call to the current policy to generate a sample from the given
        problem.
        """
        # eval might only want to get a few
        messages = [{"role": "user", "content": problem.problem}]
        if self.system_msg:
            messages = [{"role": "system", "content": self.system_msg}] + messages
        result = await client.chat.completions.create(
            messages=messages,
            model=self.vllm_model_name,
            logprobs=include_logprobs,
            max_tokens=max_seq_len,
            max_completion_tokens=max_seq_len,
            temperature=self.sampling_params.temperature,
            top_p=self.sampling_params.top_p,
            n=n_samples,
            extra_body={
                "top_k": self.sampling_params.top_k,
                "skip_special_tokens": False,
                "add_generation_prompt": True,
                "return_token_ids": True,
                "return_tokens_as_token_ids": True,
            },
        )
        return Sample.from_chat_completion(problem, result)

    async def _generate_completions(
        self,
        problems: list[Problem],
        max_seq_len: int,
        n_completions: int = None,
        logprobs=True,
    ):
        # we structure it without condensing here to avoid GC issues
        async with AsyncOpenAI(base_url=f"{self.vllm_url}/v1", timeout=None) as client:
            tasks = [
                self._generate_samples_for_problem(
                    client,
                    problem,
                    n_samples=n_completions,
                    include_logprobs=logprobs,
                    max_seq_len=max_seq_len,
                )
                for problem in problems
            ]
            return await asyncio.gather(*tasks)

    @requires_vllm_ready
    def generate_completions(
        self,
        # TODO: replace with a more lightweight datatype
        problems: list[Problem],
        max_seq_len: int,
        only_run_on_main=False,
        n_completions: int | None = None,
        include_logprobs: bool | None = None,
    ) -> list[Sample]:
        """
        Runs inference against the current server based on the given dataset
        and returns the result.
        """

        include_logprobs = True if include_logprobs is None else include_logprobs
        n_completions = (
            self.hparams.group_size if n_completions is None else n_completions
        )

        # so we want to loop through all of our samples in the dataset
        # basically we want to take a batch of samples and inititate a network request,
        # creating async tasks to later re-obtain the result

        if only_run_on_main and dist.get_rank() != 0:
            results = []
        else:
            results = asyncio.run(
                self._generate_completions(
                    problems,
                    max_seq_len,
                    n_completions=n_completions,
                    logprobs=include_logprobs,
                )
            )
        return results

    def policy_optimize_step(self) -> Tensor:
        # take an optimization step
        gradnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self._policy_has_changed = True
        self.optimizer.zero_grad()
        return gradnorm

    def save_checkpoint(self, suffix: str | None = None):
        if not self.output_dir:
            log_rank_0("no output dir provided, skipping checkpoint")
            return

        save_dir = os.path.join(self.output_dir)
        if suffix:
            save_dir = os.path.join(self.output_dir, suffix)

        os.makedirs(suffix, exist_ok=True)
        log_rank_0(f"saving checkpoint at {save_dir:!}")
        self.save_model(save_dir, self.model, self.tokenizer, torch.float32)
