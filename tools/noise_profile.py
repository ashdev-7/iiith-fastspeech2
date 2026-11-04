"""Measure the acoustic model's noise floor for clarity.denoise (writes noise_profile.json).

The profile is the median log-mel of the model's own pause frames when it reads clarity.NOISE_TEXTS
(neutral sentences, not test sentences) with the test voice and the same GV post-filter as inference.

  python tools/noise_profile.py --config English/config.yaml --data_root D --checkpoint en_model.pth.tar \
      --speaker svarah_8a8e2655 --gv_ratio Results/English/gv_ratio.json --gv_strength 1.0 \
      --out Results/English/noise_profile.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fastspeech2"))

import clarity  # noqa: E402
import inference as inf  # noqa: E402
import postfilter  # noqa: E402
from text import text_to_sequence  # noqa: E402
from utils.config import load_configs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    for a in ("--config", "--checkpoint", "--speaker", "--out"):
        ap.add_argument(a, required=True)
    ap.add_argument("--data_root")
    ap.add_argument("--gv_ratio")
    ap.add_argument("--gv_strength", type=float, default=0.0)
    a = ap.parse_args()
    configs = load_configs(a.config, a.data_root)
    lang = configs[0]["preprocessing"]["text"]["language"]
    pre = configs[0]["path"]["preprocessed_path"]
    spk = torch.tensor([json.load(open(os.path.join(pre, "speakers.json"), encoding="utf-8"))[a.speaker]]).to(inf.device)
    model, _ = inf.load_model(configs, a.checkpoint)
    gv = postfilter.load_ratio(a.gv_ratio) if a.gv_ratio and a.gv_strength else None

    def synth(phones):
        seq = text_to_sequence("{" + " ".join(phones) + "}", [], lang)
        with torch.no_grad():
            o = model(spk, torch.tensor([seq]).to(inf.device), torch.tensor([len(seq)]).to(inf.device), len(seq))
        n = int(o[9][0])
        mel = o[1][0, :n]
        if gv is not None:
            mel = postfilter.apply(mel, gv, a.gv_strength)
        return mel.cpu().numpy(), o[5][0, :len(seq)].cpu().numpy().astype(int)

    prof, n = clarity.estimate_noise_profile(synth, inf.frontend, lang)
    out = {"profile": [round(float(v), 4) for v in prof], "pause_frames": int(n), "texts": clarity.NOISE_TEXTS,
           "speaker": a.speaker, "checkpoint": os.path.basename(a.checkpoint),
           "gv_strength": a.gv_strength if gv is not None else 0.0, "digital_silence": clarity.SIL,
           "mean_db_above_silence": round(float(20 / np.log(10) * np.mean(prof - clarity.SIL)), 1)}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(a.out, "pause frames", n, "mean", out["mean_db_above_silence"], "dB above digital silence")


if __name__ == "__main__":
    main()
