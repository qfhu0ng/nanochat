"""
AIME 2024/2025 evaluation.
American Invitational Mathematics Examination — a prestigious math competition
where all answers are integers in the range [0, 999].

Datasets:
- AIME 2024: HuggingFaceH4/aime_2024 (30 problems, commit 2024-03-14)
- AIME 2025: MathArena/aime_2025 (30 problems, commit 2025-02-13)

Example usage:
    python -m scripts.chat_eval -i sft -a AIME-2024
    python -m scripts.chat_eval -i sft -a AIME-2025
    python -m scripts.chat_eval -i sft -a "AIME-2024|AIME-2025" -n 8 -t 0.7
"""

import re
from datasets import load_dataset
from tasks.common import Task


# =============================================================================
# Answer extraction (4-layer priority)
# =============================================================================

# Priority 1: LaTeX \boxed{n} format (common in math competition outputs)
BOXED_RE = re.compile(r"\\boxed\{(\d+)\}")
# Priority 2: #### n marker (GSM8K convention, learned during SFT)
MARKER_RE = re.compile(r"####\s*(\d+)")
# Priority 3: "final answer is n" or "the answer is n" (natural language)
FINAL_ANSWER_RE = re.compile(r"(?:final|the)\s+answer\s+is\s*:?\s*(\d+)", re.IGNORECASE)
# Priority 4: last standalone integer 0-999 (fallback)
STANDALONE_INT_RE = re.compile(r"(?<!\d)(\d{1,3})(?!\d)")


def extract_answer(text):
    """
    Extract the numerical answer from a completion string.
    Priority: \\boxed{n} > #### n > "answer is n" > last standalone integer.
    Returns an int in [0, 999] or None.
    """
    if not isinstance(text, str):
        return None

    # Priority 1: \boxed{n}
    match = BOXED_RE.search(text)
    if match:
        val = int(match.group(1))
        if 0 <= val <= 999:
            return val

    # Priority 2: #### n
    match = MARKER_RE.search(text)
    if match:
        val = int(match.group(1))
        if 0 <= val <= 999:
            return val

    # Priority 3: "final answer is n" / "the answer is n"
    match = FINAL_ANSWER_RE.search(text)
    if match:
        val = int(match.group(1))
        if 0 <= val <= 999:
            return val

    # Priority 4: last standalone integer in [0, 999]
    matches = STANDALONE_INT_RE.findall(text)
    for m in reversed(matches):
        val = int(m)
        if 0 <= val <= 999:
            return val

    return None


class AIME(Task):
    """
    AIME evaluation task. Loads AIME 2024 or 2025 problems from HuggingFace.
    Each problem is a math competition question with an integer answer in [0, 999].
    """

    def __init__(self, year, **kwargs):
        super().__init__(**kwargs)
        assert year in [2024, 2025], f"AIME year must be 2024 or 2025, got {year}"
        self.year = year

        if year == 2024:
            self.ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
        else:
            self.ds = load_dataset("MathArena/aime_2025", split="train")

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        problem = row["problem"]
        answer = int(row["answer"])

        # Prompt instructs the model to use #### format (consistent with GSM8K SFT training)
        prompt = (
            f"{problem}\n\n"
            "Note: This is an AIME problem. The answer is an integer between 0 and 999. "
            "Show your work, then give your final answer after ####."
        )
        solution = f"The answer is #### {answer}"

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": solution},
        ]
        return {"messages": messages}

    def evaluate(self, conversation, assistant_response):
        """
        Compare predicted answer to ground truth.
        Returns 1 if correct, 0 otherwise.
        Answers must be integers in [0, 999] to be considered valid.
        """
        gt_text = conversation["messages"][-1]["content"]
        gt_answer = extract_answer(gt_text)
        pred_answer = extract_answer(assistant_response)
        if gt_answer is None or pred_answer is None:
            return 0
        return int(pred_answer == gt_answer)
