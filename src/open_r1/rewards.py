# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reward functions for GRPO training."""

import asyncio
import json
import math
import re
from functools import partial, update_wrapper
from typing import Callable, Dict, Literal, Optional

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from .utils.code_providers import get_provider
from .utils.competitive_programming import (
    SubtaskResult,
    add_includes,
    get_morph_client_from_env,
    get_piston_client_from_env,
)
from .utils.competitive_programming import patch_code as cf_patch_code
from .utils.competitive_programming import score_submission as cf_score_submission
from .utils.competitive_programming import score_subtask


import re
from typing import List, Dict, Optional, Tuple, Any

# Assumed available from your env:
# parse, verify, LatexExtractionConfig, NormalizationConfig

# ---------- Helpers ----------

_TUPLE_RE = re.compile(
    r"""
    ^\s*
    \(?
      \s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))   # x
      \s*,\s*
      ([+-]?(?:\d+(?:\.\d+)?|\.\d+))      # y
    \s*\)?
    \s*$
    """,
    re.VERBOSE,
)

_NUMBER_RE = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)")

def _build_extraction_cfg():
    # Keep this identical for gold and answer
    return [
        LatexExtractionConfig(
            normalization_config=NormalizationConfig(
                nits=False,
                malformed_operators=False,
                basic_latex=True,   # accept x^2, \frac, etc.
                equations=False,    # allow bare expressions (no '=' required)
                boxed="all",
                units=True,         # tolerate \text{cm}, etc.
            ),
            boxed_match_priority=0,
            try_extract_without_anchor=True,  # allow plain expressions
        )
    ]

def _strip_think(text: str) -> str:
    # Remove <think> ... </think> blocks and typical fenced code
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    return text

def _normalize_units(text: str) -> str:
    """
    Convert malformed unit patterns like `7{inch}` -> `7 \text{inch}` (LaTeXish),
    but also keep a pure-numeric candidate around later.
    """
    # 12{inch} or 12{cm} -> 12 \text{inch}
    text = re.sub(r"(\d)\s*\{([A-Za-z]+)\}", r"\1 \\text{\2}", text)
    return text

def _maybe_wrap_boxed(s: str) -> str:
    """If it's a bare math-ish expression, wrap to nudge the extractor."""
    t = s.strip()
    if not t or r"\boxed" in t or "$" in t or r"\(" in t or r"\[" in t:
        return s
    # contains only typical math tokens?
    if re.fullmatch(r"[-+*/^0-9xX().,\sA-Za-z]+", t):
        return rf"\boxed{{{t}}}"
    return s

def _extract_anchored_answer(text: str) -> Optional[str]:
    """
    Try common anchors in LLM outputs to pull a concise answer string.
    Returns the captured answer WITHOUT trailing punctuation.
    """
    patterns = [
        r"ANSWER\s*:\s*(.+)$",
        r"Final\s+Answer\s*:\s*(.+)$",
        r"The\s+final\s+answer\s+is\s*:\s*(.+)$",
        r"Therefore[, ]+the\s+answer\s+is\s*[:\-]?\s*(.+)$",
        r"Thus[, ]+the\s+answer\s+is\s*[:\-]?\s*(.+)$",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            ans = m.group(1).strip()
            # remove trailing sentence enders
            ans = re.sub(r"[.\s]+$", "", ans)
            return ans
    return None

def _parse_tuple(s: str) -> Optional[Tuple[float, float]]:
    m = _TUPLE_RE.match(s.strip())
    if not m:
        return None
    return (float(m.group(1)), float(m.group(2)))

def _tuple_equal(a: str, b: str, tol: float = 1e-9) -> Optional[bool]:
    ta = _parse_tuple(a)
    tb = _parse_tuple(b)
    if ta is None or tb is None:
        return None
    return (abs(ta[0]-tb[0]) <= tol) and (abs(ta[1]-tb[1]) <= tol)

def _sympy_equal(a: str, b: str) -> Optional[bool]:
    """Optional symbolic equivalence for expressions/polynomials."""
    try:
        from sympy import sympify, Eq
    except Exception:
        return None
    aa = a.replace("^", "**").strip()
    bb = b.replace("^", "**").strip()
    try:
        return bool(Eq(sympify(aa), sympify(bb)))
    except Exception:
        return None

def _last_number(text: str) -> Optional[str]:
    nums = list(_NUMBER_RE.finditer(text))
    return nums[-1].group(0) if nums else None

def _first_boxed(text: str) -> Optional[str]:
    m = re.search(r"\\boxed\s*\{([^}]*)\}", text)
    return m.group(1).strip() if m else None

# ---------- Main reward ----------

def accuracy_reward(
    completions: List[List[Dict[str, str]]],
    solution: List[str],
    **kwargs
) -> List[Optional[float]]:
    """
    Binary reward:
      - 1.0 if the model's answer matches the gold.
      - 0.0 if verifiably wrong.
      - None if unverifiable (skip).
    Strategy:
      1) Clean & normalize text (strip <think>, fix units).
      2) Try to extract an anchored or boxed answer string from the completion.
      3) Run structured parse/verify with identical config for gold & answer.
      4) Fallbacks: tuple compare; SymPy equivalence; final numeric token match.
    """
    extraction_cfg = _build_extraction_cfg()

    rewards: List[Optional[float]] = []
    for completion, gold in zip(completions, solution):
        raw = completion[0]["content"]

        # --- sanitize/normalize ---
        content = _normalize_units(_strip_think(raw))
        # Prefer an anchored/boxed snippet if present; else keep whole content
        candidate = (
            _extract_anchored_answer(content)
            or _first_boxed(content)
            or content
        )

        # Also prepare boxed-boosted sources for the parser
        gold_src = _maybe_wrap_boxed(gold)
        ans_src  = _maybe_wrap_boxed(candidate)

        # --- primary: structured parse + verify ---
        gold_parsed = parse(gold_src, extraction_config=extraction_cfg, extraction_mode="first_match")
        if gold_parsed:
            answer_parsed = parse(ans_src, extraction_config=extraction_cfg, extraction_mode="first_match")
            try:
                reward = float(verify(gold_parsed, answer_parsed))
                rewards.append(reward)
                continue
            except Exception as e:
                print(f"[verify failed] {e}\n answer: {answer_parsed}\n gold: {gold_parsed}")

        # --- fallback A: exact tuple equality (common in geometry/coords) ---
        t_eq = _tuple_equal(gold, candidate)
        if t_eq is True:
            rewards.append(1.0); continue
        if t_eq is False:
            rewards.append(0.0); continue

        # --- fallback B: symbolic equivalence (polynomials/expressions) ---
        s_eq = _sympy_equal(gold, candidate)
        if s_eq is True:
            rewards.append(1.0); continue
        if s_eq is False:
            rewards.append(0.0); continue

        # --- fallback C: match last numeric token if gold is numeric ---
        # (Useful when answer is "The width is 7 feet" and gold is "7")
        if _NUMBER_RE.fullmatch(gold.strip()):
            last_num = _last_number(candidate)
            if last_num is not None:
                rewards.append(1.0 if last_num == gold.strip() else 0.0)
                continue

        # Could not verify; skip this item
        print("Failed to parse/verify example.\nGold:", gold, "\nCandidate:", candidate)
        rewards.append(None)

    return rewards



def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    pattern = r"<think>.*</think>.*\\boxed\{.*\}.*"
    # pattern = r".*\\boxed\{.*\}.*"
    # pattern = r"<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
            # count =+ 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
            # count =+ 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://huggingface.co/papers/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(
                sol,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float, language: str = "en"):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://huggingface.co/papers/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    language: Language of the text, defaults to `en`. Used to choose the way to split the text into n-grams.
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    if language == "en":

        def zipngram(text: str, ngram_size: int):
            words = text.lower().split()
            return zip(*[words[i:] for i in range(ngram_size)]), words

    elif language == "zh":
        from transformers.utils.import_utils import _is_package_available

        if not _is_package_available("jieba"):
            raise ValueError("Please install jieba to use Chinese language")

        def zipngram(text: str, ngram_size: int):
            import jieba

            seg_list = list(jieba.cut(text))
            return zip(*[seg_list[i:] for i in range(ngram_size)]), seg_list

    else:
        raise ValueError(
            f"Word splitting for language `{language}` is not yet implemented. Please implement your own zip-ngram function."
        )

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            ngram_array, words = zipngram(completion, ngram_size)

            if len(words) < ngram_size:
                rewards.append(0.0)
                continue

            for ng in ngram_array:
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def _init_event_loop():
    """Initialize or get the current event loop."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def ioi_code_reward(completions, test_batch_size: int = 1, provider_type: str = "piston", **kwargs) -> list[float]:
    """Reward function that evaluates IOI problems using a specified execution client.

    Assumes the dataset has the same format as hf.co/datasets/open-r1/ioi

    Args:
        completions: List of model completions to evaluate
        test_batch_size: Evaluate these many test cases in parallel, then check if any of them failed (0 score):
                       if so stop evaluating; otherwise continue with the next batch of test cases.
        provider_type: The execution provider to use (default: "piston"). Supported values: "piston", "morph"
        **kwargs: Additional arguments passed from the dataset
    """
    # Get the appropriate client based on provider_type
    if provider_type == "morph":
        execution_client = get_morph_client_from_env()
    else:
        # for info on setting up piston workers, see slurm/piston/README.md
        execution_client = get_piston_client_from_env()

    code_snippets = [
        # note: grading is automatically skipped if no code is extracted
        add_includes(extract_code(completion[-1]["content"], "cpp"), problem_id)
        for completion, problem_id in zip(completions, kwargs["id"])
    ]

    async def run_catch_exceptions(task):
        try:
            return await task
        except Exception as e:
            print(f"Error from {provider_type} worker: {e}")
            return SubtaskResult()

    problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]

    loop = _init_event_loop()
    evals = [
        loop.create_task(
            run_catch_exceptions(
                score_subtask(
                    execution_client,
                    problem_data,
                    code,
                    test_batch_size=test_batch_size,
                )
            )
        )
        for problem_data, code in zip(problems_data, code_snippets)
    ]
    results = loop.run_until_complete(asyncio.gather(*evals))

    return [result.score for result in results]


def cf_code_reward(
    completions,
    test_batch_size: int = 1,
    patch_code: bool = False,
    scoring_mode: Literal["pass_fail", "partial", "weighted_sum"] = "weighted_sum",
    **kwargs,
) -> list[float]:
    """Reward function that evaluates Codeforces problems using Piston+our CF package.

    Assumes the dataset has the same format as hf.co/datasets/open-r1/codeforces (verifiable-prompts subset)

    test_batch_size: evaluate these many test cases in parallel, then check if any of them failed (0 score): if so stop evaluating; otherwise continue with the next batch of test cases.
    """
    # for info on setting up piston workers, see slurm/piston/README.md
    piston_client = get_piston_client_from_env()

    languages = kwargs["language"] if "language" in kwargs else [None] * len(completions)
    code_snippets = [
        # note: grading is automatically skipped if a problem has no tests
        cf_patch_code(extract_code(completion[-1]["content"], language), language)
        if patch_code
        else extract_code(completion[-1]["content"], language)
        for completion, language in zip(completions, languages)
    ]

    async def run_catch_exceptions(task):
        try:
            return await task
        except Exception as e:
            print(f"Error from Piston worker: {e}")
            return None

    # load problem data. undo separating kwargs by column
    problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]

    loop = _init_event_loop()
    evals = [
        loop.create_task(
            run_catch_exceptions(
                cf_score_submission(
                    piston_client,
                    problem_data,
                    code,
                    test_batch_size=test_batch_size,
                    scoring_mode=scoring_mode,
                    submission_language=problem_data.get("language", None),
                )
            )
        )
        for problem_data, code in zip(problems_data, code_snippets)
    ]
    results = loop.run_until_complete(asyncio.gather(*evals))

    return results


def extract_code(completion: str, language: str | None = "python") -> str:
    if language is None:
        return ""
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def binary_code_reward(
    completions,
    num_parallel: int = 2,
    provider_type: str = "e2b",
    enforce_same_language: bool = False,
    **kwargs,
) -> list[float]:
    rewards = code_reward(
        completions,
        num_parallel=num_parallel,
        provider_type=provider_type,
        enforce_same_language=enforce_same_language,
        **kwargs,
    )
    BINARY_THRESHOLD = 0.99

    output = []
    for reward in rewards:
        if reward is None:
            output.append(None)
        else:
            output.append(1.0 if reward > BINARY_THRESHOLD else 0.0)

    return output


def code_reward(
    completions,
    num_parallel: int = 2,
    provider_type: str = "e2b",
    enforce_same_language: bool = False,
    **kwargs,
) -> list[float]:
    """Reward function that evaluates code snippets using a code execution provider.

    Assumes the dataset contains a `verification_info` column with test cases.

    Args:
        completions: List of model completions to evaluate
        num_parallel: Number of parallel code executions (default: 2)
        provider_type: Which code execution provider to use (default: "e2b")
        enforce_same_language: If True, verify all problems use the same language (default: False)
        **kwargs: Additional arguments passed to the verification
    """
    evaluation_script_template = """
    import subprocess
    import json

    def evaluate_code(code, test_cases):
        passed = 0
        total = len(test_cases)
        exec_timeout = 5

        for case in test_cases:
            process = subprocess.run(
                ["python3", "-c", code],
                input=case["input"],
                text=True,
                capture_output=True,
                timeout=exec_timeout
            )

            if process.returncode != 0:  # Error in execution
                continue

            output = process.stdout.strip()

            # TODO: implement a proper validator to compare against ground truth. For now we just check for exact string match on each line of stdout.
            all_correct = True
            for line1, line2 in zip(output.split('\\n'), case['output'].split('\\n')):
                all_correct = all_correct and line1.strip() == line2.strip()

            if all_correct:
                passed += 1

        success_rate = (passed / total)
        return success_rate

    code_snippet = {code}
    test_cases = json.loads({test_cases})

    evaluate_code(code_snippet, test_cases)
    """

    code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
    verification_info = kwargs["verification_info"]

    template = evaluation_script_template

    scripts = [
        template.format(code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"])))
        for code, info in zip(code_snippets, verification_info)
    ]

    language = verification_info[0]["language"]

    if enforce_same_language:
        all_same_language = all(v["language"] == language for v in verification_info)
        if not all_same_language:
            raise ValueError("All verification_info must have the same language", verification_info)

    execution_provider = get_provider(
        provider_type=provider_type,
        num_parallel=num_parallel,
        **kwargs,
    )

    return execution_provider.execute_scripts(scripts, ["python"] * len(scripts))


def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """

    def code_format_reward(completions, **kwargs):
        # if there is a language field, use it instead of the default language. This way we can have mixed language training.
        languages = kwargs["language"] if "language" in kwargs else [language] * len(completions)

        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [
            re.match(
                rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{sample_language}.*?```.*?\n</answer>$",
                content,
                re.DOTALL | re.MULTILINE,
            )
            for content, sample_language in zip(completion_contents, languages)
        ]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward


def get_soft_overlong_punishment(max_completion_len, soft_punish_cache):
    """
    Reward function that penalizes overlong completions. It is used to penalize overlong completions,
    but not to reward shorter completions. Reference: Eq. (13) from the DAPO paper (https://huggingface.co/papers/2503.14476)

    Args:
        max_completion_len: Maximum length of the completion
        soft_punish_cache: Minimum length of the completion. If set to 0, no minimum length is applied.
    """

    def soft_overlong_punishment_reward(completion_ids: list[list[int]], **kwargs) -> list[float]:
        """Reward function that penalizes overlong completions."""
        rewards = []
        for ids in completion_ids:
            completion_length = len(ids)
            if completion_length <= max_completion_len - soft_punish_cache:
                rewards.append(0.0)
            elif max_completion_len - soft_punish_cache < completion_length <= max_completion_len:
                rewards.append((max_completion_len - soft_punish_cache - completion_length) / soft_punish_cache)
            else:
                rewards.append(-1.0)
        return rewards

    return soft_overlong_punishment_reward


def get_reward_funcs(script_args) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": update_wrapper(
            partial(
                code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                provider_type=script_args.code_provider,
                enforce_same_language=getattr(script_args, "enforce_same_language", False),
            ),
            code_reward,
        ),
        "binary_code": update_wrapper(
            partial(
                binary_code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                provider_type=script_args.code_provider,
                enforce_same_language=getattr(script_args, "enforce_same_language", False),
            ),
            binary_code_reward,
        ),
        "ioi_code": update_wrapper(
            partial(
                ioi_code_reward,
                test_batch_size=script_args.code_eval_test_batch_size,
                provider_type=getattr(script_args, "ioi_provider", "piston"),
            ),
            ioi_code_reward,
        ),
        "cf_code": update_wrapper(
            partial(
                cf_code_reward,
                test_batch_size=script_args.code_eval_test_batch_size,
                scoring_mode=script_args.code_eval_scoring_mode,
            ),
            cf_code_reward,
        ),
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
        "soft_overlong_punishment": get_soft_overlong_punishment(
            max_completion_len=script_args.max_completion_len,
            soft_punish_cache=script_args.soft_punish_cache,
        ),
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs
