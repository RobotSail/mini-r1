"""
Modular reward functions for GRPO training.

To change the reward function used during training, either:
1. Change DEFAULT_REWARD_FN at the bottom of this file
2. Pass --reward-fn=<name> to the CLI (if supported)

To add a new reward function:
1. Create a function that takes (response: str, ground_truth: Any) -> RolloutScore
2. Optionally add it to REWARD_REGISTRY for CLI access
"""

from __future__ import annotations

import re
import functools
from typing import Protocol, Any
import random

from type_defs import RolloutScore


class RewardFn(Protocol):
    """Protocol for reward functions."""

    def __call__(self, response: str, ground_truth: Any) -> RolloutScore: ...


# =============================================================================
# Parsing Utilities
# =============================================================================


@functools.lru_cache(maxsize=1)
def answer_pattern() -> re.Pattern:
    return re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


@functools.lru_cache(maxsize=1)
def think_pattern() -> re.Pattern:
    return re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def parse_number(text: str) -> int | float:
    """
    Parse a string into a number with proper sanitization.
    Handles common formatting like commas, dollar signs, percentages, etc.
    Raises ValueError if parsing fails.
    """
    # Strip whitespace
    text = text.strip()

    # Remove common currency symbols and formatting
    text = text.replace("$", "")
    text = text.replace("€", "")
    text = text.replace("£", "")
    text = text.replace("¥", "")
    text = text.replace("₹", "")

    # Remove percentage signs (but keep the numeric value)
    text = text.replace("%", "")

    # Remove commas (thousand separators)
    text = text.replace(",", "")

    # Remove underscores (sometimes used as thousand separators)
    text = text.replace("_", "")

    # Remove spaces (sometimes used in number formatting)
    text = text.replace(" ", "")

    # Handle parentheses notation for negative numbers (accounting format)
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]

    # Strip again after all replacements
    text = text.strip()

    # Check if we have any digits left
    if not any(c.isdigit() for c in text):
        raise ValueError(f"No digits found in text after sanitization: {text}")

    # Try to parse as float or int
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError as e:
        raise ValueError(f"Failed to parse '{text}' as number: {e}")


def extract_answer_from_tags(response: str) -> int | float | None:
    """
    Extract the numeric answer from <answer>...</answer> tags.
    Returns None if no valid answer found.
    """
    matches = answer_pattern().findall(response)
    parsed_nums: list[int | float] = []

    for match in matches:
        try:
            parsed_nums.append(parse_number(match))
        except ValueError:
            continue

    # Return the last valid answer (model's final answer)
    return parsed_nums[-1] if parsed_nums else None


# =============================================================================
# Reward Components (building blocks)
# =============================================================================


def random_reward(mean: float = 0, std: float = 1, min_r=0.01):
    r = random.normalvariate(mean, std)
    return max(min_r, r)


def format_reward_think_answer(response: str) -> float:
    """
    Reward for following the <think>...</think><answer>...</answer> format.

    Returns:
        0.0 - No valid format
        0.1 - Has thinking content
        1.0 - Has thinking AND exactly one answer after </think>
    """
    # Must start with <think>
    if not response.startswith("<think>"):
        return 0.0

    # Must have closing </think>
    if "</think>" not in response:
        return 0.0

    # Check for thinking content
    thoughts = think_pattern().findall(response)
    if len(thoughts) < 1:
        return 0.0

    # lets do random rewards here

    reward = random_reward(mean=0.1, std=0.05, min_r=0.01)

    # Small reward for having thinking content
    if len(thoughts[0].strip()) > 0:
        reward += random_reward(mean=0.1, std=0.05, min_r=0.01)

    # Check for exactly one answer after the thinking section
    post_think = response.split("</think>")[-1]
    answers = answer_pattern().findall(post_think)
    if len(answers) == 1:
        reward += random_reward(mean=1.0, std=0.1, min_r=0.9)

    return reward


def accuracy_reward_numeric(response: str, ground_truth: int | float) -> tuple[float, bool, bool]:
    """
    Check if the response contains the correct numeric answer.

    Looks for the answer in <think>...</think> tags (legacy behavior).

    Returns:
        (reward, is_correct, is_parsable)
    """
    thoughts = think_pattern().findall(response)
    if len(thoughts) < 1:
        return (0.0, False, False)

    # Use the last thought block
    assumed_answer = thoughts[-1]

    try:
        answer = parse_number(assumed_answer)
        is_parsable = True
    except ValueError:
        return (0.0, False, False)

    is_correct = answer == ground_truth
    reward = random_reward(1.0, 0.3, min_r=0.9) if is_correct else 0.0

    return (reward, is_correct, is_parsable)


def accuracy_reward_answer_tags(response: str, ground_truth: int | float) -> tuple[float, bool, bool]:
    """
    Check if the response contains the correct numeric answer in <answer> tags.

    Returns:
        (reward, is_correct, is_parsable)
    """
    answer = extract_answer_from_tags(response)

    if answer is None:
        return (0.0, False, False)

    is_correct = answer == ground_truth
    reward = 1.0 if is_correct else 0.1  # slight reward for correct format

    return (reward, is_correct, True)


# =============================================================================
# Composite Reward Functions
# =============================================================================


def math_with_thinking(response: str, ground_truth: Any) -> RolloutScore:
    """
    Default reward: format compliance + accuracy.

    - Format reward (0.1 for thinking, 1.0 for proper format)
    - Accuracy reward (1.0 for correct answer)

    Total possible: 2.0 (1.0 format + 1.0 accuracy)
    """
    format_r = format_reward_think_answer(response)
    acc_r, is_correct, is_parsable = accuracy_reward_numeric(response, ground_truth)

    return RolloutScore(
        reward=format_r + acc_r,
        is_correct=is_correct,
        is_parsable=is_parsable,
    )


def accuracy_only(response: str, ground_truth: Any) -> RolloutScore:
    """
    Pure accuracy reward - no format requirements.
    Uses <answer> tags for parsing.
    """
    acc_r, is_correct, is_parsable = accuracy_reward_answer_tags(response, ground_truth)

    return RolloutScore(
        reward=acc_r,
        is_correct=is_correct,
        is_parsable=is_parsable,
    )


def format_only(response: str, ground_truth: Any) -> RolloutScore:
    """
    Pure format compliance reward - ignores accuracy.
    Useful for testing format learning before adding accuracy.
    """
    format_r = format_reward_think_answer(response)

    return RolloutScore(
        reward=format_r,
        is_correct=False,
        is_parsable=False,
    )


def strict_format_with_accuracy(response: str, ground_truth: Any) -> RolloutScore:
    """
    Strict format: only reward accuracy if format is correct.
    Format must be perfect (reward=1.0) before accuracy counts.
    """
    format_r = format_reward_think_answer(response)

    if format_r < 1.0:
        # Format not correct, no reward at all
        return RolloutScore(reward=0.0, is_correct=False, is_parsable=False)

    # Format is correct, now check accuracy
    acc_r, is_correct, is_parsable = accuracy_reward_answer_tags(response, ground_truth)

    return RolloutScore(
        reward=format_r + acc_r,
        is_correct=is_correct,
        is_parsable=is_parsable,
    )


def strict_answer_only(response: str, ground_truth: Any) -> RolloutScore:
    """
    Strict format without thinking requirement: rewards for <answer> tags presence and accuracy.

    Rewards:
        - 0.1 for having <answer>...</answer> tags (format compliance)
        - 1.0 additional for correct answer (total 1.1 if correct)
    """
    FORMAT_REWARD = 0.1
    ACCURACY_REWARD = 1.0

    # Check if answer tags are present
    if not ("<answer>" in response and "</answer>" in response):
        return RolloutScore(reward=0.0, is_correct=False, is_parsable=False)

    # Tags present - give format reward
    reward = FORMAT_REWARD

    # Try to extract and check answer
    answer = extract_answer_from_tags(response)
    if answer is None:
        # Tags present but can't parse a number - still get format reward
        return RolloutScore(reward=reward, is_correct=False, is_parsable=False)

    # Number was parsable
    is_correct = answer == ground_truth
    if is_correct:
        reward += ACCURACY_REWARD

    return RolloutScore(
        reward=reward,
        is_correct=is_correct,
        is_parsable=True,
    )


# =============================================================================
# Registry and Default
# =============================================================================

REWARD_REGISTRY: dict[str, RewardFn] = {
    "math_with_thinking": math_with_thinking,
    "accuracy_only": accuracy_only,
    "format_only": format_only,
    "strict_format_with_accuracy": strict_format_with_accuracy,
    "r1-zero": math_with_thinking,
    "gsm8k": strict_answer_only,
}


def get_reward_fn(name: str) -> RewardFn:
    """Get a reward function by name. Raises KeyError if not found."""
    if name not in REWARD_REGISTRY:
        available = ", ".join(REWARD_REGISTRY.keys())
        raise KeyError(f"Unknown reward function '{name}'. Available: {available}")
    return REWARD_REGISTRY[name]


# =============================================================================
# DEFAULT - Change this to swap reward behavior
# =============================================================================

DEFAULT_REWARD_FN: RewardFn = REWARD_REGISTRY["gsm8k"]
