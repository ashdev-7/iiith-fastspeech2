"""Where does the quality go? Natural recording vs vocoder vs acoustic model, on held-out data.

For up to --n validation utterances of the test voice it renders four versions of the same sentence:
  natural  the recording (22.05 kHz, as ingested)
  copy     ground-truth mel -> HiFi-GAN                 (vocoder loss only)
  oracle   FastSpeech 2 with ground-truth duration/pitch/energy -> HiFi-GAN   (+ decoder loss)
  free     FastSpeech 2 free-running -> HiFi-GAN         (+ variance-predictor loss) = what we ship
and measures each with Praat (tools/acoustic_analysis.py) plus the long-term spectrum in bands, relative
to the natural recording. A band that is several dB below "natural" in "oracle" but not in "copy" is
over-smoothing by the decoder (the usual cause of muffled FastSpeech 2 output); a loss already in
"copy" is the vocoder.

Outputs: <out>/diagnosis.json (no audio from the dataset is written to the repository; the wavs go to
<out>/wavs/ for listening and are packaged separately by the Kaggle job).
"""
import argparse
import json
import os
import statistics
import sys

import numpy as np
import torch
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))
import acoustic_analysis as A  # noqa: E402
import parselmouth  # noqa: E402

from dataset import Dataset  # noqa: E402
from evaluate_val import oracle_mel, synthesize  # noqa: E402
from inference import device, load_model  # noqa: E402
from utils.config import load_configs  # noqa: E402
from utils.model import get_vocoder, vocoder_infer  # noqa: E402

BANDS = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 6000), (6000, 8000), (8000, 11025)]
CONDITIONS = ["natural", "copy", "oracle", "free"]


def trim(x, sr, top_db=40.0):
    import librosa
    y, _ = librosa.effects.trim(x.astype(np.float32), top_db=top_db, frame_length=1024, hop_length=256)
    return y


def ltas_bands(x, sr):
    """Mean power per band over speech frames (frames within 40 dB of the loudest), dB re total power."""
    n, hop = 1024, 256
    if len(x) < n:
        return None
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop] * np.hanning(n)
    p = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    e = p.sum(1)
    p = p[e > e.max() * 1e-4]
    f = np.fft.rfftfreq(n, 1 / sr)
    tot = p.sum()
    return [float(10 * np.log10(p[:, (f >= lo) & (f < hi)].sum() / tot + 1e-20)) for lo, hi in BANDS]


def analyse(x, sr):
    snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=sr)
    m = A.sentence_level(snd)
    m["bands_db"] = ltas_bands(x, sr)
    return m


def rank_speakers(a, configs, model, vocoder):
    """Every training speaker: clarity of its natural recordings and of copy-synthesis (3 utterances each).
    clarity = mean 1-4 kHz band share (dB) + HNR/2; the 1-4 kHz band carries formant and consonant cues."""
    pre_cfg, model_cfg, train_cfg = configs
    sr = pre_cfg["preprocessing"]["audio"]["sampling_rate"]
    raw = pre_cfg["path"]["raw_path"]
    ds = Dataset("train.txt", pre_cfg, train_cfg, sort=False, drop_last=False)
    by_spk = {}
    for i in range(len(ds)):
        by_spk.setdefault(ds.speaker[i], []).append(i)
    out = {}
    for spk, ids in sorted(by_spk.items()):
        rec = {"utterances_in_train": len(ids), "natural": [], "copy": []}
        for i in ids[:3]:
            s = ds[i]
            _, nat = wavfile.read(os.path.join(raw, spk, s["id"] + ".wav"))
            mel_gt = torch.from_numpy(s["mel"].T).float().unsqueeze(0).to(device)
            cp = vocoder_infer(mel_gt, vocoder, model_cfg, pre_cfg)[0]
            for name, w in (("natural", nat), ("copy", cp)):
                w = trim(w.astype(np.float32) / 32768.0, sr)
                b = ltas_bands(w, sr)
                hnr = A.sentence_level(parselmouth.Sound(w.astype(np.float64), sampling_frequency=sr)).get("hnr_db")
                if b is not None and hnr is not None:
                    rec[name].append({"band_1_4k_db": 10 * np.log10(10 ** (b[2] / 10) + 10 ** (b[3] / 10)), "hnr_db": hnr})
        for name in ("natural", "copy"):
            v = rec.pop(name)
            if v:
                rec[name + "_band_1_4k_db"] = round(float(np.mean([x["band_1_4k_db"] for x in v])), 2)
                rec[name + "_hnr_db"] = round(float(np.mean([x["hnr_db"] for x in v])), 2)
        if "copy_band_1_4k_db" in rec:
            rec["clarity"] = round(rec["copy_band_1_4k_db"] + rec["copy_hnr_db"] / 2, 2)
        out[spk] = rec
        print("rank", spk, rec, flush=True)
    ranked = sorted((k for k in out if "clarity" in out[k]), key=lambda k: -out[k]["clarity"])
    json.dump({"rule": rank_speakers.__doc__, "ranked": ranked, "speakers": out},
              open(os.path.join(a.out_dir, "speaker_clarity.json"), "w", encoding="utf-8"), indent=1)
    return ranked


def main(a):
    configs = load_configs(a.config, a.data_root)
    pre_cfg, model_cfg, train_cfg = configs
    sr = pre_cfg["preprocessing"]["audio"]["sampling_rate"]
    raw = pre_cfg["path"]["raw_path"]
    model, step = load_model(configs, a.checkpoint)
    vocoder = get_vocoder(model_cfg, device, a.vocoder)
    os.makedirs(a.out_dir, exist_ok=True)
    if a.rank_speakers:
        rank_speakers(a, configs, model, vocoder)
    ds = Dataset("val.txt", pre_cfg, train_cfg, sort=False, drop_last=False)
    idx = [i for i in range(len(ds)) if ds.speaker[i] == a.speaker][: a.n]
    if len(idx) < 5:
        raise SystemExit("only {} validation utterances for {}".format(len(idx), a.speaker))
    os.makedirs(os.path.join(a.out_dir, "wavs"), exist_ok=True)
    rows = []
    for i in idx:
        s = ds[i]
        sr_n, nat = wavfile.read(os.path.join(raw, ds.speaker[i], s["id"] + ".wav"))
        assert sr_n == sr, (sr_n, sr)
        nat = nat.astype(np.float32) / 32768.0
        mel_gt = torch.from_numpy(s["mel"].T).float().unsqueeze(0).to(device)
        copy_w = vocoder_infer(mel_gt, vocoder, model_cfg, pre_cfg)[0].astype(np.float32) / 32768.0
        mel_or = torch.from_numpy(oracle_mel(model, s).T).float().unsqueeze(0).to(device)
        orac_w = vocoder_infer(mel_or, vocoder, model_cfg, pre_cfg)[0].astype(np.float32) / 32768.0
        free_w = synthesize(model, vocoder, configs, s["speaker"], s["text"])["wav"].astype(np.float32) / 32768.0
        row = {"id": s["id"], "text": ds.raw_text[i]}
        for name, w in zip(CONDITIONS, (nat, copy_w, orac_w, free_w)):
            w = trim(w, sr)
            row[name] = analyse(w, sr)
            if len(rows) < a.save_wavs:
                wavfile.write(os.path.join(a.out_dir, "wavs", "{}_{}.wav".format(s["id"], name)), sr,
                              (np.clip(w, -1, 1) * 32767).astype(np.int16))
        rows.append(row)
        print("diagnose {}/{}".format(len(rows), len(idx)), flush=True)

    keys = ["hnr_db", "hf_share_db", "cog_hz", "jitter_pct", "shimmer_pct", "f0_range_st", "voiced_frac", "intensity_db"]
    summary = {}
    for c in CONDITIONS:
        summary[c] = {k: round(statistics.median(r[c][k] for r in rows if r[c].get(k) is not None), 2) for k in keys}
        bands = np.array([r[c]["bands_db"] for r in rows if r[c]["bands_db"]])
        summary[c]["bands_db"] = [round(float(v), 2) for v in bands.mean(0)]
    for c in CONDITIONS[1:]:
        summary[c]["bands_minus_natural_db"] = [round(x - y, 2) for x, y in zip(summary[c]["bands_db"],
                                                                                summary["natural"]["bands_db"])]
    out = {"speaker": a.speaker, "checkpoint_step": step, "utterances": len(rows),
           "bands_hz": BANDS, "conditions": CONDITIONS, "summary": summary,
           "per_utterance": [{k: v for k, v in r.items() if k != "text"} for r in rows],
           "note": __doc__}
    json.dump(out, open(os.path.join(a.out_dir, "diagnosis.json"), "w", encoding="utf-8"), indent=1)
    for c in CONDITIONS:
        print(c, json.dumps(summary[c]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--vocoder", default=None, help="fine-tuned HiFi-GAN generator (default: universal)")
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--save_wavs", type=int, default=6)
    ap.add_argument("--rank_speakers", action="store_true", help="also rank all training speakers by clarity")
    main(ap.parse_args())
