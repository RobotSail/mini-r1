import typer


def preview_tokenization(dataset, tokenizer):
    """Preview tokenized messages before training."""
    typer.secho("\n" + "=" * 60, fg=typer.colors.BRIGHT_YELLOW)
    typer.secho("  TOKENIZATION PREVIEW", fg=typer.colors.BRIGHT_YELLOW, bold=True)
    typer.secho("=" * 60, fg=typer.colors.BRIGHT_YELLOW)

    # Get a sample from the dataset
    preview_sample = next(iter(dataset.shuffle().iter(1)))
    preview_messages = preview_sample["messages"][0]

    typer.secho("\nOriginal messages:", fg=typer.colors.BRIGHT_WHITE)
    for msg in preview_messages:
        typer.secho(f"  [{msg['role']}]: {msg['content']}", fg=typer.colors.WHITE)

    # Tokenize the messages
    tokenized = tokenizer.apply_chat_template(
        conversation=preview_messages,
        return_tensors="pt",
        add_generation_prompt=False,
    )

    typer.secho(
        f"\nTokenized input_ids shape: {tokenized.shape}", fg=typer.colors.BRIGHT_WHITE
    )
    typer.secho(f"Number of tokens: {tokenized.numel()}", fg=typer.colors.BRIGHT_WHITE)
    typer.secho(f"\nToken IDs: {tokenized[0].tolist()}", fg=typer.colors.CYAN)

    # Decode back to show what the model sees
    decoded = tokenizer.decode(tokenized[0], skip_special_tokens=False)
    typer.secho(f"\nDecoded (with special tokens):", fg=typer.colors.BRIGHT_WHITE)
    typer.secho(f"{decoded}", fg=typer.colors.GREEN)

    typer.secho("\n" + "=" * 60 + "\n", fg=typer.colors.BRIGHT_YELLOW)


def display_scorecard(rollouts: list, epoch: int, epochs: int):
    # Calculate and display epoch scorecard
    total_rollouts = sum(len(sample.rollouts) for sample in rollouts)
    parsable_rollouts = sum(
        sum(1 for r in sample.rollouts if r.is_parsable) for sample in rollouts
    )
    correct_rollouts = sum(
        sum(1 for r in sample.rollouts if r.is_correct) for sample in rollouts
    )

    parsable_pct = (
        (parsable_rollouts / total_rollouts * 100) if total_rollouts > 0 else 0
    )
    correct_pct = (correct_rollouts / total_rollouts * 100) if total_rollouts > 0 else 0
    accuracy_pct = (
        (correct_rollouts / parsable_rollouts * 100) if parsable_rollouts > 0 else 0
    )

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
