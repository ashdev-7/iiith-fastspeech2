import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The vendored reference code uses top-level imports ("from text import ...").
sys.path.insert(0, os.path.join(ROOT, "fastspeech2"))


@pytest.fixture(scope="session")
def test_items():
    with open(os.path.join(ROOT, "tests", "test_sentences.json"), encoding="utf-8") as f:
        return {item["id"]: item for item in json.load(f)["items"]}
