"""Patched reference preprocessor on a synthetic utterance: manifest-driven split, MFA empty
intervals kept as "sp", durations matching mel frames. Needs torch/librosa/pyworld/tgt; skipped
where they are missing (runs on Kaggle before feature extraction)."""
import csv
import json
import os

import pytest

for _mod in ("numpy", "torch", "tgt", "librosa", "pyworld", "sklearn", "soundfile"):
    pytest.importorskip(_mod)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import tgt  # noqa: E402

from preprocessor.preprocessor import Preprocessor  # noqa: E402

SR, HOP = 22050, 256


def _config(root):
    return {
        "path": {"raw_path": f"{root}/raw", "split_path": f"{root}/split", "preprocessed_path": f"{root}/pre"},
        "preprocessing": {
            "audio": {"sampling_rate": SR, "max_wav_value": 32768.0},
            "stft": {"filter_length": 1024, "hop_length": HOP, "win_length": 1024},
            "mel": {"n_mel_channels": 80, "mel_fmin": 0, "mel_fmax": 8000},
            "pitch": {"feature": "phoneme_level", "normalization": True},
            "energy": {"feature": "phoneme_level", "normalization": True},
        },
    }


def _utterance(root, speaker, utt, f0):
    os.makedirs(f"{root}/raw/{speaker}", exist_ok=True)
    t = np.arange(int(1.2 * SR)) / SR
    wav = 0.3 * np.sin(2 * np.pi * f0 * t) * (t > 0.1) * (t < 1.1)
    sf.write(f"{root}/raw/{speaker}/{utt}.wav", wav.astype(np.float32), SR)
    with open(f"{root}/raw/{speaker}/{utt}.lab", "w", encoding="utf-8") as f:
        f.write("పులి అడవి\n")
    tg = tgt.TextGrid()
    # MFA style: leading/trailing silence and the inter-word pause are EMPTY intervals (not written).
    phones = tgt.IntervalTier(0, 1.2, "phones")
    for s, e, p in [(0.1, 0.3, "p"), (0.3, 0.5, "u"), (0.6, 0.8, "a"), (0.8, 1.1, "i")]:
        phones.add_interval(tgt.Interval(s, e, p))
    tg.add_tier(phones)
    os.makedirs(f"{root}/pre/TextGrid/{speaker}", exist_ok=True)
    tgt.io.write_to_file(tg, f"{root}/pre/TextGrid/{speaker}/{utt}.TextGrid", format="long")


def _manifest(root, split, rows):
    os.makedirs(f"{root}/split", exist_ok=True)
    with open(f"{root}/split/{split}.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["utt_id", "speaker", "group", "duration", "model_text", "split"])
        for spk, utt in rows:
            w.writerow([utt, spk, "", "1.2", "x", split])


def test_preprocessor_on_synthetic_data(tmp_path):
    root = str(tmp_path)
    _utterance(root, "te_male", "te_00001", 120)
    _utterance(root, "te_female", "te_00002", 220)
    _utterance(root, "te_female", "te_00003", 230)   # on disk but in no manifest: must be ignored
    _manifest(root, "train", [("te_male", "te_00001")])
    _manifest(root, "val", [("te_female", "te_00002"), ("te_female", "te_00009")])  # 00009: no TextGrid

    Preprocessor(_config(root)).build_from_path()
    pre = f"{root}/pre"
    train = open(f"{pre}/train.txt", encoding="utf-8").read().splitlines()
    val = open(f"{pre}/val.txt", encoding="utf-8").read().splitlines()
    assert [line.split("|")[0] for line in train] == ["te_00001"]
    assert [line.split("|")[0] for line in val] == ["te_00002"]
    # The pause between the words is kept as "sp"; leading/trailing silence is trimmed.
    assert train[0].split("|")[2] == "{p u sp a i}"
    assert train[0].split("|")[3] == "పులి అడవి"
    assert json.load(open(f"{pre}/speakers.json", encoding="utf-8")) == {"te_female": 0, "te_male": 1}
    summary = json.load(open(f"{pre}/preprocess_summary.json", encoding="utf-8"))
    assert summary["skipped"]["no_textgrid"] == ["te_00009"]

    for spk, utt in (("te_male", "te_00001"), ("te_female", "te_00002")):
        dur = np.load(f"{pre}/duration/{spk}-duration-{utt}.npy")
        mel = np.load(f"{pre}/mel/{spk}-mel-{utt}.npy")
        pitch = np.load(f"{pre}/pitch/{spk}-pitch-{utt}.npy")
        energy = np.load(f"{pre}/energy/{spk}-energy-{utt}.npy")
        assert len(dur) == 5 and dur.sum() == mel.shape[0]
        assert mel.shape[1] == 80 and len(pitch) == len(energy) == len(dur)
        assert np.all(np.isfinite(pitch)) and np.all(np.isfinite(energy))
