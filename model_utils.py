from torch.distributed.fsdp import MixedPrecisionPolicy
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
import torch


def prepare_model(model_save_dir: str):
    """
    prepares the OLMO model for training
    """
    os.makedirs(model_save_dir, exist_ok=True)

    # the final pretrained checkpoint
    model_name = "allenai/Olmo-3-1025-7B"
    revision = "stage1-step1413814"

    # load assets
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)

    # we will be saving the model in float16
    llm = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, dtype=torch.float16)

    # olmo tokenizer is already prepared, we just need to define the chat template
    with open("chat-template.jinja", "r") as infile:
        tokenizer.chat_template = infile.read()

    # ensure these are correct
    assert tokenizer.pad_token_id == llm.config.pad_token_id
    assert tokenizer.eos_token_id == llm.config.eos_token_id

    # save
    tokenizer.save_pretrained(model_save_dir)
    llm.config.save_pretrained(model_save_dir)
    llm.save_pretrained(model_save_dir)


def wrap_model_with_fsdp2(model, training_model=True):
    """
    Here we initialize the fsdp2 device mesh and wrap the model
    """

    mp_policy = MixedPrecisionPolicy(
        # compute forward in bf16
        param_dtype=torch.bfloat16 if training_model else torch.float16,  # we want high precision where possible
        reduce_dtype=torch.float32,
    )

    # Disable HuggingFace cache if present
    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception as e:
            print(f"WARNING: Failed to disable HuggingFace cache for model {model.__class__.__name__}: {e}")

    # select layers
    layers = model.model.layers

    # apply activation checkpointing
    if training_model:
        for i, layer in enumerate(layers):
            layers[i] = ptd_checkpoint_wrapper(layer, preserve_rng=True)

    # now we wrap wiht fsdp2
