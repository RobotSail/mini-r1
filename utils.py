import inspect
import typer
import subprocess
import sys
import torch
import os
from datetime import timedelta
import torch.distributed as dist
import logging


class StreamablePopen(subprocess.Popen):
    """
    Provides a way of reading stdout and stderr line by line.
    """

    def __init__(self, output_file, *args, **kwargs):
        # remove the stderr and stdout from kwargs
        kwargs.pop("stderr", None)
        kwargs.pop("stdout", None)
        self.output_file = output_file

        super().__init__(*args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs)

    def listen(self, prefix: str = ""):
        with open(self.output_file, "wb") as full_log_file:
            line_buffer = bytearray()
            while True:
                byte = self.stdout.read(1)
                if byte:
                    line_buffer.extend(byte)
                    full_log_file.write(byte)

                    # Check if we've hit a newline
                    if byte == b"\n":
                        # Decode and print the complete line with prefix
                        line_text = line_buffer.decode("utf-8", "ignore")
                        sys.stdout.write(prefix + line_text)
                        sys.stdout.flush()
                        line_buffer.clear()
                else:
                    # No more data - print any remaining buffered content
                    if line_buffer:
                        line_text = line_buffer.decode("utf-8", "ignore")
                        sys.stdout.write(prefix + line_text)
                        sys.stdout.flush()
                    break


# def preview_tokenization(dataset, tokenizer):
#     """Preview tokenized messages before training."""
#     typer.secho("\n" + "=" * 60, fg=typer.colors.BRIGHT_YELLOW)
#     typer.secho("  TOKENIZATION PREVIEW", fg=typer.colors.BRIGHT_YELLOW, bold=True)
#     typer.secho("=" * 60, fg=typer.colors.BRIGHT_YELLOW)

#     # Get a sample from the dataset
#     preview_sample = next(iter(dataset.shuffle().iter(1)))
#     preview_messages = preview_sample["messages"][0]

#     typer.secho("\nOriginal messages:", fg=typer.colors.BRIGHT_WHITE)
#     for msg in preview_messages:
#         typer.secho(f"  [{msg['role']}]: {msg['content']}", fg=typer.colors.WHITE)

#     # Tokenize the messages
#     tokenized = tokenizer.apply_chat_template(
#         conversation=preview_messages,
#         return_tensors="pt",
#         add_generation_prompt=False,
#     )

#     typer.secho(f"\nTokenized input_ids shape: {tokenized.shape}", fg=typer.colors.BRIGHT_WHITE)
#     typer.secho(f"Number of tokens: {tokenized.numel()}", fg=typer.colors.BRIGHT_WHITE)
#     typer.secho(f"\nToken IDs: {tokenized[0].tolist()}", fg=typer.colors.CYAN)

#     # Decode back to show what the model sees
#     decoded = tokenizer.decode(tokenized[0], skip_special_tokens=False)
#     typer.secho(f"\nDecoded (with special tokens):", fg=typer.colors.BRIGHT_WHITE)
#     typer.secho(f"{decoded}", fg=typer.colors.GREEN)

#     typer.secho("\n" + "=" * 60 + "\n", fg=typer.colors.BRIGHT_YELLOW)


def streaming_loop(logfile: str, command: list[str], *args, **kwargs):
    interrupt = None
    try:
        process = StreamablePopen(logfile, command, *args, **kwargs)
        process.listen(prefix="[vllm-server] ")
    except KeyboardInterrupt as e:
        print("Training subprocess interrupted by user.")
        interrupt = e
    except Exception as e:
        print("Unexpected exception received during distributed training", e)
        interrupt = e
    finally:
        if "process" not in locals() or process is None:
            return

        # wait for the process to exit so we can properly read the exit code
        process.wait(timeout=60)
        process_code = process.poll()
        failure = process_code != 0

        if not failure:
            print("Operation completed successfully! 🎉")
        else:
            print(f"Training subprocess has not exited yet. Sending SIGTERM. Process code: {process_code}")

        process.terminate()
        try:
            print("Waiting for process to exit, 60s...")
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            print("Training subprocess did not terminate before timeout, sending SIGKILL.")
            process.kill()

        if interrupt:
            raise interrupt
        if failure:
            raise RuntimeError(
                "Suffered a failure during distributed training. Please see the training logs for more context."
            )


def display_scorecard(rollouts: list, epoch: int, epochs: int):
    # Calculate and display epoch scorecard
    total_rollouts = sum(len(sample.rollouts) for sample in rollouts)
    parsable_rollouts = sum(sum(1 for r in sample.rollouts if r.is_parsable) for sample in rollouts)
    correct_rollouts = sum(sum(1 for r in sample.rollouts if r.is_correct) for sample in rollouts)

    parsable_pct = (parsable_rollouts / total_rollouts * 100) if total_rollouts > 0 else 0
    correct_pct = (correct_rollouts / total_rollouts * 100) if total_rollouts > 0 else 0
    accuracy_pct = (correct_rollouts / parsable_rollouts * 100) if parsable_rollouts > 0 else 0

    typer.secho("\n" + "=" * 60, fg=typer.colors.BRIGHT_CYAN)
    typer.secho(
        f"  EPOCH {epoch + 1}/{epochs} SCORECARD",
        fg=typer.colors.BRIGHT_CYAN,
        bold=True,
    )
    typer.secho("=" * 60, fg=typer.colors.BRIGHT_CYAN)
    typer.secho(f"  Total Rollouts:     {total_rollouts}", fg=typer.colors.WHITE)
    typer.secho(
        f"  Parsable:           {parsable_rollouts}/{total_rollouts} ({parsable_pct:.1f}%)",
        fg=typer.colors.YELLOW if parsable_pct < 90 else typer.colors.GREEN,
    )
    typer.secho(
        f"  Correct:            {correct_rollouts}/{total_rollouts} ({correct_pct:.1f}%)",
        fg=typer.colors.GREEN if correct_pct > 50 else typer.colors.RED,
    )
    typer.secho(
        f"  Accuracy (of parsable): {correct_rollouts}/{parsable_rollouts} ({accuracy_pct:.1f}%)",
        fg=typer.colors.BRIGHT_GREEN
        if accuracy_pct > 70
        else typer.colors.BRIGHT_YELLOW
        if accuracy_pct > 40
        else typer.colors.BRIGHT_RED,
    )
    typer.secho("=" * 60 + "\n", fg=typer.colors.BRIGHT_CYAN)


def get_caller(num_frames=1):
    frame = inspect.currentframe().f_back
    for _ in range(num_frames - 1):
        frame = frame.f_back
    file_name = frame.f_code.co_filename
    line_number = frame.f_lineno
    return f"In {file_name}, line {line_number}"


def log_rank_0(msg, include_caller=False, rank=None, to_print=True):
    if rank is None:
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        rank = local_rank if dist.is_initialized() else 0
    if rank <= 0:
        if include_caller:
            msg = f"{get_caller(num_frames=2)}: {msg}"
        if to_print:
            print(msg)
        else:
            logging.info(msg)


def check_distributed_is_synchronized():
    """
    This function runs a simple check to verify that torch.distributed
    is functioning properly and all processes are synchronized.
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    t = torch.tensor([1]).to(device, torch.int32)

    # Here, every process group increments the counter
    # so the total amount should equal the world size.
    # all_reduce here is functionally equivalent to `dist.barrier`
    dist.all_reduce(t, op=dist.ReduceOp.SUM)

    # We should see that all GPUs add the value up to 8
    assert t.item() == dist.get_world_size(), "❌ Error: distributed check failed"


def check_distributed_is_evenly_configured():
    """
    DDP, FSDP1, and FSDP2 do not support uneven world-size configurations,
    and therefore neither do our distributed computing algorithms (e.g. distributed SVD init).
    PyTorch/torchrun should be enforcing this by default, but we double-check this here
    in case PyTorch ever changes their APIs or stops enforcing it.
    """
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

    # check that world_size is cleanly divisible by device count here:
    if world_size % local_world_size != 0:
        raise ValueError(
            f"world_size ({world_size}) is not cleanly divisible by local_world_size ({local_world_size}). Each node must have the same number of GPUs."
        )

    device = torch.device("cuda", local_rank)
    max_local_rank_seen = torch.tensor([local_rank], dtype=torch.int32, device=device)
    dist.all_reduce(max_local_rank_seen, op=dist.ReduceOp.MAX)
    if max_local_rank_seen[0] != local_world_size - 1:
        raise ValueError(
            f"max_local_rank_seen ({max_local_rank_seen[0]}) is not equal to local_world_size ({local_world_size}). Each node must have the same number of GPUs."
        )


def init_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=180), device_id=device)
    # NOTE(osilkin): PyTorch wants us to avoid this API in favor of setting the device explicitly
    # through `init_process_group`, but without setting this, FSDP2 will shard the
    # entire model onto the first GPU. I haven't yet figured out a solution to this.
    torch.cuda.set_device(local_rank)
    check_distributed_is_synchronized()
    check_distributed_is_evenly_configured()
    log_rank_0("✅ Torch distributed appears to be functioning correctly")

    # Log distributed configuration details
    world_size = dist.get_world_size()
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    num_nodes = world_size // local_world_size
    log_rank_0(
        f"📊 Distributed Configuration:\n"
        f"   • World Size: {world_size}\n"
        f"   • Number of Nodes: {num_nodes}\n"
        f"   • GPUs per Node: {local_world_size}"
    )

    torch.distributed.barrier()


def destroy_distributed_environment():
    # wait for checkpoints to show up, once training is complete we tear it down
    dist.barrier()
    log_rank_0("Training complete 😀, tearing down distributed environment")
    dist.destroy_process_group()
