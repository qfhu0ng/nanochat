"""
Schema validation tests for safety SFT data.

Verifies that dev/safety_conversations.jsonl is compatible with CustomJSON:
- Each line is a valid JSON array of message objects
- Messages alternate user/assistant roles
- Each message has 'role' and 'content' string fields
- At least 2 messages per conversation
- At least 500 total conversations
"""

import json
import os
import pytest

SAFETY_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "dev", "safety_conversations.jsonl")
SAFETY_META_PATH = os.path.join(os.path.dirname(__file__), "..", "dev", "safety_conversations.meta.jsonl")


def load_conversations():
    """Load all conversations from the safety data file."""
    if not os.path.exists(SAFETY_DATA_PATH):
        pytest.skip(f"Safety data not found at {SAFETY_DATA_PATH} (run gen_safety_data.py first)")
    conversations = []
    with open(SAFETY_DATA_PATH, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            conversations.append((line_num, json.loads(line)))
    return conversations


def load_metadata():
    """Load all metadata from the sidecar file."""
    if not os.path.exists(SAFETY_META_PATH):
        pytest.skip(f"Metadata not found at {SAFETY_META_PATH}")
    meta = []
    with open(SAFETY_META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                meta.append(json.loads(line))
    return meta


class TestSafetyDataSchema:
    """Validate that each conversation matches CustomJSON expected format."""

    def test_file_exists(self):
        assert os.path.exists(SAFETY_DATA_PATH), f"Safety data file not found: {SAFETY_DATA_PATH}"

    def test_minimum_count(self):
        conversations = load_conversations()
        assert len(conversations) >= 500, f"Expected >= 500 conversations, got {len(conversations)}"

    def test_each_line_is_valid_json_array(self):
        conversations = load_conversations()
        for line_num, messages in conversations:
            assert isinstance(messages, list), f"Line {line_num}: expected list, got {type(messages)}"

    def test_minimum_two_messages(self):
        conversations = load_conversations()
        for line_num, messages in conversations:
            assert len(messages) >= 2, f"Line {line_num}: expected >= 2 messages, got {len(messages)}"

    def test_alternating_roles(self):
        conversations = load_conversations()
        for line_num, messages in conversations:
            for i, msg in enumerate(messages):
                expected_role = "user" if i % 2 == 0 else "assistant"
                assert msg["role"] == expected_role, (
                    f"Line {line_num}, message {i}: expected role '{expected_role}', got '{msg['role']}'"
                )

    def test_message_fields(self):
        conversations = load_conversations()
        for line_num, messages in conversations:
            for i, msg in enumerate(messages):
                assert "role" in msg, f"Line {line_num}, message {i}: missing 'role'"
                assert "content" in msg, f"Line {line_num}, message {i}: missing 'content'"
                assert isinstance(msg["content"], str), (
                    f"Line {line_num}, message {i}: content must be string, got {type(msg['content'])}"
                )

    def test_no_empty_content(self):
        conversations = load_conversations()
        for line_num, messages in conversations:
            for i, msg in enumerate(messages):
                assert msg["content"].strip(), f"Line {line_num}, message {i}: empty content"

    def test_customjson_compatible(self):
        """Load via CustomJSON to verify end-to-end compatibility."""
        if not os.path.exists(SAFETY_DATA_PATH):
            pytest.skip("Safety data not generated yet")
        from tasks.customjson import CustomJSON
        dataset = CustomJSON(filepath=SAFETY_DATA_PATH)
        assert len(dataset) >= 500
        # Spot check a few examples
        for i in range(min(5, len(dataset))):
            example = dataset[i]
            assert "messages" in example
            assert len(example["messages"]) >= 2


class TestSafetyMetadata:
    """Validate the metadata sidecar file."""

    def test_meta_exists(self):
        if not os.path.exists(SAFETY_DATA_PATH):
            pytest.skip("Safety data not generated yet")
        assert os.path.exists(SAFETY_META_PATH), "Meta file should exist alongside final data"

    def test_meta_line_count_matches(self):
        conversations = load_conversations()
        meta = load_metadata()
        assert len(meta) == len(conversations), (
            f"Meta has {len(meta)} lines but data has {len(conversations)} lines"
        )

    def test_all_8_topics_covered(self):
        meta = load_metadata()
        topics = {m["topic"] for m in meta}
        expected_topics = {
            "violence_harm", "illegal_activity", "privacy_violation",
            "discrimination_hate", "misinformation", "child_safety",
            "self_harm", "malicious_software",
        }
        assert topics == expected_topics, f"Missing topics: {expected_topics - topics}"

    def test_meta_has_required_fields(self):
        meta = load_metadata()
        required = {"topic", "dynamic", "num_turns"}
        for i, m in enumerate(meta):
            for field in required:
                assert field in m, f"Meta line {i}: missing field '{field}'"
