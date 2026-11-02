"""tools/acoustic_analysis.py: word timing from predicted durations covers every phone of every test sentence."""
import json
import os
import sys

import pytest

pytest.importorskip("parselmouth")
pytest.importorskip("numpy")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import acoustic_analysis as A  # noqa: E402


@pytest.mark.parametrize("lang", ["English", "Telugu"])
def test_words_cover_all_phones(lang):
    base = os.path.join(ROOT, "Results", lang)
    trace = json.load(open(os.path.join(base, "frontend_trace.json"), encoding="utf-8"))
    for sid, tr in trace.items():
        raw = json.load(open(os.path.join(base, "variance_raw", sid + ".json"), encoding="utf-8"))
        words = A.words_with_times(raw, tr)
        assert words[0]["t0"] == 0
        assert abs(words[-1]["t1"] - sum(raw["duration_ms"]) / 1000) < 0.05, sid
        assert all(w["t1"] >= w["t0"] for w in words)
