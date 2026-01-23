"""
Custom vLLM worker extension for weight reloading.
Based on Prime RL's FileSystemWeightUpdateWorker implementation.
See: https://github.com/PRIME-RL/PRIME/blob/main/src/prime_rl/inference/vllm/worker/filesystem.py

CRITICAL: vLLM's _get_weights_iterator() caches weights from initial load and does NOT
re-read from disk. We bypass this by loading directly from safetensors files.
"""

from typing import TYPE_CHECKING
import os
import torch

from torch.nn import Module
from safetensors.torch import load_file
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.logger import init_logger

# type hints without extending at runtime (required by vLLM worker extension)
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object

logger = init_logger("mini_r1.vllm_worker_extension")


class FileSystemWeightUpdateWorker(Worker):
    """vLLM worker extension for updating weights in-place using shared filesystem."""

    def update_weights(self, weight_path: str) -> None:
        """Update weights from a specified path containing a HF-compatible checkpoint.

        CRITICAL: vLLM's _get_weights_iterator() caches weights from initial load.
        We bypass this by loading directly from safetensors files.
        """
        logger.info(f"update_weights called with path: {weight_path}")

        # get vLLM model runner and unwrap the model
        model_runner = self.model_runner

        # access the raw model (unwrap from CUDAGraphWrapper/UBatchWrapper if needed)
        if hasattr(model_runner.model, "runnable"):
            model = model_runner.model.runnable
        elif hasattr(model_runner.model, "unwrap"):
            model = model_runner.model.unwrap()
        else:
            model = model_runner.model

        assert isinstance(model, Module), f"Expected nn.Module, got {type(model)}"

        # Build a map of model parameter names to their tensors
        model_params = dict(model.named_parameters())

        # Load ALL weights directly from safetensors files (BYPASS vLLM's broken caching)
        safetensors_files = sorted([f for f in os.listdir(weight_path) if f.endswith(".safetensors")])

        # Collect all weights from all shards
        all_weights = {}
        for sf_file in safetensors_files:
            sf_path = os.path.join(weight_path, sf_file)
            try:
                tensors = load_file(sf_path)
                all_weights.update(tensors)
            except Exception as e:
                logger.error(f"Error loading {sf_file}: {e}")
                raise

        logger.info(f"Loaded {len(all_weights)} weights from {len(safetensors_files)} files")

        # Copy weights directly to model parameters
        # Handle vLLM's fused weight naming (qkv_proj, gate_up_proj)
        copied_count = 0
        fused_count = 0

        with torch.no_grad():
            for param_name, param in model_params.items():
                # Check for direct match first
                if param_name in all_weights:
                    weight = all_weights[param_name]
                    param.data.copy_(weight.to(param.device, param.dtype))
                    copied_count += 1
                    continue

                # Check for fused qkv_proj weight
                if "qkv_proj" in param_name:
                    # Need to combine q_proj, k_proj, v_proj
                    weight_loader = getattr(param, "weight_loader", None)
                    if weight_loader is not None:
                        for source_suffix, shard_id in [("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v")]:
                            source_name = param_name.replace("qkv_proj", source_suffix)
                            if source_name in all_weights:
                                weight = all_weights[source_name]
                                weight_loader(param, weight.to(param.device, param.dtype), shard_id)
                        fused_count += 1
                    continue

                # Check for fused gate_up_proj weight
                if "gate_up_proj" in param_name:
                    # Need to combine gate_proj, up_proj
                    weight_loader = getattr(param, "weight_loader", None)
                    if weight_loader is not None:
                        for source_suffix, shard_id in [("gate_proj", 0), ("up_proj", 1)]:
                            source_name = param_name.replace("gate_up_proj", source_suffix)
                            if source_name in all_weights:
                                weight = all_weights[source_name]
                                weight_loader(param, weight.to(param.device, param.dtype), shard_id)
                        fused_count += 1
                    continue

        logger.info(f"Copied {copied_count} direct weights, {fused_count} fused weights")

        # CRITICAL: process weights after loading (handles quantization, attention init, etc.)
        device = next(model.parameters()).device
        process_weights_after_loading(model, model_runner.model_config, device)

        logger.info("Weight update complete")
