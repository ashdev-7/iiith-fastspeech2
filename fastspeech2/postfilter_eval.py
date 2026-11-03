"""Estimate the GV post-filter ratio on training clips of the test voice, then measure its effect on
held-out validation clips of that voice at several strengths.

Writes <out>/gv_ratio.json (ratio per mel bin; goes to Results/<Lang>/) and <out>/postfilter_eval.json:
per strength, the variance ratio and frame-to-frame flux relative to the natural mel, MCD (DTW) to the
natural mel, and Praat HNR / jitter / shimmer of the vocoded audio. A few wavs per strength go to
<out>/wavs/ for listening (dataset-derived text, our synthesis only).
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

import eval_metrics as M  # noqa: E402
import postfilter  # noqa: E402
from dataset import Dataset  # noqa: E402
from evaluate_val import synthesize  # noqa: E402
from inference import device, load_model  # noqa: E402
from utils.config import load_configs  # noqa: E402
from utils.model import get_vocoder, vocoder_infer  # noqa: E402


def flux(mel):
    return float(np.mean(np.abs(np.diff(mel, axis=0))))


def main(a):
    configs = load_configs(a.config, a.data_root)
    pre_cfg, model_cfg, train_cfg = configs
    sr = pre_cfg["preprocessing"]["audio"]["sampling_rate"]
    model, step = load_model(configs, a.checkpoint)
    vocoder = get_vocoder(model_cfg, device)
    os.makedirs(os.path.join(a.out_dir, "wavs"), exist_ok=True)

    tr = Dataset("train.txt", pre_cfg, train_cfg, sort=False, drop_last=False)
    idx = [i for i in range(len(tr)) if tr.speaker[i] == a.speaker][: a.n_train]
    gt, pr = [], []
    for i in idx:
        s = tr[i]
        r = synthesize(model, vocoder, configs, s["speaker"], s["text"])
        if r["mel_len"]:
            gt.append(s["mel"])
            pr.append(r["mel"])
    ratio = postfilter.estimate_ratio(gt, pr)
    json.dump({"speaker": a.speaker, "checkpoint_step": step, "train_utterances": len(gt), "ratio": ratio,
               "note": postfilter.__doc__}, open(os.path.join(a.out_dir, "gv_ratio.json"), "w"), indent=1)
    print("ratio per bin (min/median/max): {:.2f} / {:.2f} / {:.2f}".format(min(ratio), statistics.median(ratio), max(ratio)))

    va = Dataset("val.txt", pre_cfg, train_cfg, sort=False, drop_last=False)
    vidx = [i for i in range(len(va)) if va.speaker[i] == a.speaker][: a.n_val]
    res = {str(st): [] for st in a.strengths}
    for n, i in enumerate(vidx):
        s = va[i]
        r = synthesize(model, vocoder, configs, s["speaker"], s["text"])
        if not r["mel_len"]:
            continue
        g = s["mel"]
        c_gt = M.mel_cepstra(g)
        for st in a.strengths:
            m = postfilter.apply(r["mel"], ratio, st)
            w = vocoder_infer(torch.from_numpy(m.T).float().unsqueeze(0).to(device), vocoder, model_cfg, pre_cfg)[0]
            snd = parselmouth.Sound(w.astype(np.float64) / 32768.0, sampling_frequency=sr)
            q = A.sentence_level(snd)
            res[str(st)].append({
                "gv_ratio": float(np.mean(postfilter.utterance_gv(m) / np.maximum(postfilter.utterance_gv(g), 1e-8))),
                "flux_ratio": flux(m) / flux(g), "mcd_dtw": M.mcd_dtw(M.mel_cepstra(m), c_gt)[0],
                "hnr_db": q.get("hnr_db"), "jitter_pct": q.get("jitter_pct"), "shimmer_pct": q.get("shimmer_pct")})
            if n < a.save_wavs:
                wavfile.write(os.path.join(a.out_dir, "wavs", "{}_gv{}.wav".format(s["id"], st)), sr, w)
        print("val {}/{}".format(n + 1, len(vidx)), flush=True)
    summary = {st: {k: round(statistics.median(x[k] for x in rows if x[k] is not None), 3) for k in rows[0]}
               for st, rows in res.items() if rows}
    json.dump({"speaker": a.speaker, "val_utterances": len(vidx), "summary": summary, "per_utterance": res},
              open(os.path.join(a.out_dir, "postfilter_eval.json"), "w"), indent=1)
    for st, v in summary.items():
        print("strength", st, v)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_train", type=int, default=40)
    ap.add_argument("--n_val", type=int, default=25)
    ap.add_argument("--save_wavs", type=int, default=4)
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    main(ap.parse_args())
