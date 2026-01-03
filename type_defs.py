import pydantic
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class Problem(pydantic.BaseModel):
    problem: str
    answer: int
    operation: str


class SamplingParams(pydantic.BaseModel):
    temperature: float
    top_p: float = 1.0
    top_k: float = 0.0
    repetition_penalty: float = 1.0
    max_tokens: int = 0
    max_new_tokens: int = 0


class Message(pydantic.BaseModel):
    role: str
    content: str


class TokenSample(pydantic.BaseModel):
    token: int
    logprob: float


class RolloutResult(pydantic.BaseModel):
    response: str
    logprobs: list[TokenSample] | None = None
    is_parsable: bool = False
    is_correct: bool = False
    reward: float = 0.0
    advantage: float = 0.0
    seed_messages: list[Message] = pydantic.Field(
        default_factory=list,
    )

    def to_dataset_format(self):
        """
        Returns a dict in a format that can be used in datasets
        """
        return {
            "advantage": self.advantage,
            "messages": self.rollout_trace_to_json_list(),
            "logprobs": [lp.model_dump() for lp in self.logprobs],
        }

    def seed_messages_to_json_list(self):
        """
        Serializes only the seed messages as a list that can be consumed
        in the expected JSON format.
        """
        return [m.model_dump() for m in self.seed_messages]

    def rollout_trace_to_json_list(self):
        """
        Serializes the full rollout trace as a list that can be consumed
        in the expected JSON format.
        """
        return self.seed_messages_to_json_list() + [{"role": "assistant", "content": self.response}]


# class RolloutData(pydantic.BaseModel):
#     logprobs: list[TokenSample]
#     response: str
#     is_parsable: bool = False
#     is_correct: bool = False


class Sample(pydantic.BaseModel):
    problem: Problem
    # system_prompt: str | None = None
    # results: list[SampleResult] = pydantic.Field(default_factory=list)
    rollouts: list[RolloutResult] = pydantic.Field(default_factory=list)


class Hyperparameters(pydantic.BaseModel):
    lr: float
    model_name: str
    max_seq_len: int
    batch_size: int
    num_rollouts: int
    epochs: int
    inner_epochs: int
    inner_batch_size: int
    eps: float = 0.1  # or 0.2 as in deepseek's original paper
    kl_penalty_strength: float = 1.0


class TrainingComponents(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    optimizer: torch.optim.Optimizer
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    train_tokenizer: PreTrainedTokenizerBase = pydantic.Field(
        metadata={
            "help": "this is a custom tokenizer used for unmasking messages during training. See instructlab.training.data_process.configure_tokenizer"
        }
    )
    hyperparams: Hyperparameters
    device: torch.device
    sampling_params: SamplingParams
