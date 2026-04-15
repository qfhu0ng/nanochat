"""
Unit tests for AIME evaluation task.
Tests answer extraction (4-layer priority), range validation, and evaluate logic.

Run: python -m pytest tests/test_aime_task.py -v
"""

import pytest
from tasks.aime import extract_answer, AIME


class TestExtractAnswer:
    """Test the 4-layer answer extraction priority."""

    # Priority 1: \boxed{n}
    def test_boxed_simple(self):
        assert extract_answer(r"The answer is \boxed{204}") == 204

    def test_boxed_zero(self):
        assert extract_answer(r"\boxed{0}") == 0

    def test_boxed_max(self):
        assert extract_answer(r"\boxed{999}") == 999

    def test_boxed_out_of_range(self):
        # \boxed{1000} is out of range, should fall through to lower priority
        assert extract_answer(r"\boxed{1000}") is None

    # Priority 2: #### n
    def test_marker_simple(self):
        assert extract_answer("The answer is #### 42") == 42

    def test_marker_no_space(self):
        assert extract_answer("####42") == 42

    def test_marker_zero(self):
        assert extract_answer("#### 0") == 0

    # Priority 3: "final answer is n" / "the answer is n"
    def test_final_answer_is(self):
        assert extract_answer("So the final answer is 7") == 7

    def test_the_answer_is(self):
        assert extract_answer("Therefore, the answer is 123") == 123

    def test_answer_is_colon(self):
        assert extract_answer("The answer is: 456") == 456

    # Priority 4: last standalone integer
    def test_last_standalone(self):
        assert extract_answer("We get 100 + 50 + 3 = 153") == 153

    def test_last_standalone_single_digit(self):
        assert extract_answer("The result is 7.") == 7

    # Priority ordering: boxed > marker > final answer > standalone
    def test_boxed_wins_over_marker(self):
        assert extract_answer(r"#### 100 but actually \boxed{200}") == 200

    def test_marker_wins_over_final(self):
        assert extract_answer("The answer is 100 #### 200") == 200

    def test_final_wins_over_standalone(self):
        assert extract_answer("We compute 50 + 50 = 100. The final answer is 200") == 200

    # Edge cases
    def test_empty_string(self):
        assert extract_answer("") is None

    def test_none_input(self):
        assert extract_answer(None) is None

    def test_no_numbers(self):
        assert extract_answer("no numbers here at all") is None

    def test_only_large_numbers(self):
        # All numbers > 999, no valid AIME answer
        assert extract_answer("The population is 1000000 and area is 5000") is None


class TestRangeValidation:
    """Test that answers outside [0, 999] are rejected."""

    def test_negative_marker_extracts_positive(self):
        # "#### -5" doesn't match the #### regex (no negative support),
        # but fallback extracts 5 which is in range. This is acceptable
        # since AIME answers are never negative.
        assert extract_answer("#### -5") == 5

    def test_1000_rejected(self):
        assert extract_answer("#### 1000") is None

    def test_zero_accepted(self):
        assert extract_answer("#### 0") == 0

    def test_999_accepted(self):
        assert extract_answer("#### 999") == 999


class TestAIMEEvaluate:
    """Test the evaluate method with mock conversations."""

    def _make_conversation(self, answer):
        return {
            "messages": [
                {"role": "user", "content": "Some math problem"},
                {"role": "assistant", "content": f"The answer is #### {answer}"},
            ]
        }

    def test_correct_answer(self):
        aime = AIME.__new__(AIME)  # skip __init__ (no HF download)
        conv = self._make_conversation(204)
        assert aime.evaluate(conv, "I computed #### 204") == 1

    def test_wrong_answer(self):
        aime = AIME.__new__(AIME)
        conv = self._make_conversation(204)
        assert aime.evaluate(conv, "I think #### 205") == 0

    def test_no_answer_in_response(self):
        aime = AIME.__new__(AIME)
        conv = self._make_conversation(204)
        assert aime.evaluate(conv, "I have no idea") == 0

    def test_boxed_format_accepted(self):
        aime = AIME.__new__(AIME)
        conv = self._make_conversation(42)
        assert aime.evaluate(conv, r"Therefore \boxed{42}") == 1

    def test_natural_language_accepted(self):
        aime = AIME.__new__(AIME)
        conv = self._make_conversation(7)
        assert aime.evaluate(conv, "After all that work, the final answer is 7") == 1


class TestAIMEDataLoading:
    """Test that AIME datasets load correctly from HuggingFace."""

    @pytest.mark.parametrize("year,expected_count", [(2024, 30), (2025, 30)])
    def test_dataset_loads(self, year, expected_count):
        task = AIME(year=year)
        assert task.num_examples() == expected_count

    @pytest.mark.parametrize("year", [2024, 2025])
    def test_example_structure(self, year):
        task = AIME(year=year)
        example = task[0]
        assert "messages" in example
        msgs = example["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert "####" in msgs[1]["content"]

    @pytest.mark.parametrize("year", [2024, 2025])
    def test_all_answers_in_range(self, year):
        task = AIME(year=year)
        for i in range(len(task)):
            example = task[i]
            answer = extract_answer(example["messages"][-1]["content"])
            assert answer is not None, f"Problem {i} has no extractable answer"
            assert 0 <= answer <= 999, f"Problem {i} answer {answer} out of range"

    def test_invalid_year(self):
        with pytest.raises(AssertionError):
            AIME(year=2023)

    @pytest.mark.parametrize("year", [2024, 2025])
    def test_eval_type(self, year):
        task = AIME(year=year)
        assert task.eval_type == "generative"
