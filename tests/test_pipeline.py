"""kaggle/pipeline.py helpers that need no Kaggle: subset selection, config merge, status, finalize."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kaggle"))
import pipeline as P  # noqa: E402


def _manifests(root):
    man = os.path.join(root, "manifests", "te")
    os.makedirs(man)
    rows = lambda spk, n, split: [{"utt_id": f"{spk}_{split}_{i}", "speaker": spk, "group": "", "duration": str(1 + i % 7),  # noqa: E731
                                   "model_text": "x", "split": split} for i in range(n)]
    P.write_tsv(os.path.join(man, "train.tsv"), rows("a", 50, "train") + rows("b", 30, "train") + rows("c", 90, "train"))
    P.write_tsv(os.path.join(man, "val.tsv"), rows("a", 10, "val") + rows("b", 10, "val"))  # c has no val
    return man


def test_subset_prefers_speakers_with_val(tmp_path):
    man = _manifests(str(tmp_path))
    info = P.subset_manifests(str(tmp_path), "te", n_train=5, n_val=2, max_speakers=2)
    assert info == {"speakers": ["a", "b"], "train": 10, "val": 4}
    train = P.read_tsv(os.path.join(man, "train.tsv"))
    assert {r["speaker"] for r in train} == {"a", "b"}
    assert max(float(r["duration"]) for r in train) == 1.0  # shortest clips first


def test_test_voice_rules(tmp_path):
    _manifests(str(tmp_path))
    assert P.test_voice(str(tmp_path), "te") == "c"          # no copy-synthesis result: most training audio
    cs = tmp_path / "cs.json"
    cs.write_text(json.dumps({"ranked": ["b", "a"]}))
    assert P.test_voice(str(tmp_path), "te", str(cs)) == "b"  # best copy-synthesis


def test_derived_config_deep_merge(tmp_path):
    pytest.importorskip("yaml")
    import shutil
    import yaml
    for d in ("Telugu", "English"):
        os.makedirs(tmp_path / d)
        shutil.copy(os.path.join(ROOT, d, "config.yaml"), tmp_path / d / "config.yaml")
    path = P.derived_config(str(tmp_path), "te", "x", {"train": {"step": {"total_step": 30}}})
    cfg = yaml.safe_load(open(path, encoding="utf-8"))
    assert cfg["train"]["step"]["total_step"] == 30
    assert cfg["train"]["step"]["log_step"] == 100           # untouched keys survive
    assert cfg["preprocess"]["preprocessing"]["text"]["language"] == "te"


def test_status_records_failures_and_continues(tmp_path):
    st = P.Status(path=str(tmp_path / "status.json"), job="t")
    with st.stage("good"):
        pass
    with st.stage("soft", fatal=False):
        raise RuntimeError("boom")
    with pytest.raises(ValueError):
        with st.stage("hard"):
            raise ValueError("x")
    data = json.load(open(tmp_path / "status.json"))
    assert data["stages"]["good"]["ok"] is True
    assert data["stages"]["soft"]["ok"] is False and "boom" in data["stages"]["soft"]["error"]
    assert data["fatal_stage"] == "hard"


def test_finalize_bundles_loose_files(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "OUT", str(tmp_path))
    for n in ("status.json", "a.json", "big.tar"):
        (tmp_path / n).write_text("{}")
    os.makedirs(tmp_path / "logs" / "deep")
    (tmp_path / "logs" / "deep" / "x.log").write_text("x")
    assert P.finalize() == ["big.tar", "details.tar", "status.json"]


def test_optional_output_never_raises():
    assert P.optional_output(["definitely-not-a-real-tool-xyz"]).startswith("[definitely-not-a-real-tool-xyz unavailable")
    assert "hi" in P.optional_output([sys.executable, "-c", "print('hi')"])
    assert P.optional_output([sys.executable, "-c", "import sys; sys.exit(3)"]).startswith("[")
