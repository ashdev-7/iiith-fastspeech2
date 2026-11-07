"""Background-noise floor, speech level and ending level of generated speech (report Section 6.4).

For every wav present in both folders (same file name), in 20 ms frames (hop 10 ms), levels in dBFS:
  quiet  : mean level of the 10 % quietest frames between the first and last frame within 40 dB of the
           loudest frame (pauses and closures inside the sentence: the audible background)
  speech : median level of the 50 % loudest frames
  end    : level of the last 20 ms of the file (does the waveform stop mid-sound?)
Typical use: model-only output (no inference-time options) vs the final output of the same sentences.

  python tools/clarity_measure.py Evaluation/English/model_only/test_generated_audios Results/English/test_generated_audios
"""
import argparse
import json
import os

import numpy as np
from scipy.io import wavfile


def levels(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float64) / 32768.0
    n, h = int(0.02 * sr), int(0.01 * sr)
    frames = np.array([x[i:i + n] for i in range(0, len(x) - n + 1, h)])
    db = 10 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
    active = np.where(db > db.max() - 40)[0]
    inner = np.sort(db[active[0]:active[-1] + 1])
    k = max(1, len(inner) // 10)
    end = 10 * np.log10(np.mean(x[-n:] ** 2) + 1e-12)
    return {"quiet_db": float(inner[:k].mean()), "speech_db": float(np.median(inner[len(inner) // 2:])),
            "end_db": float(end)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--json")
    a = ap.parse_args()
    names = sorted(set(os.listdir(a.before)) & set(os.listdir(a.after)))
    rows = {f: {"before": levels(os.path.join(a.before, f)), "after": levels(os.path.join(a.after, f))} for f in names
            if f.endswith(".wav")}
    for f, r in rows.items():
        b, c = r["before"], r["after"]
        print("{:12s} quiet {:6.1f} -> {:6.1f} dB   speech {:6.1f} -> {:6.1f} dB   end {:6.1f} -> {:6.1f} dB".format(
            f, b["quiet_db"], c["quiet_db"], b["speech_db"], c["speech_db"], b["end_db"], c["end_db"]))
    mean = {s: {k: round(float(np.mean([r[s][k] for r in rows.values()])), 1) for k in ("quiet_db", "speech_db", "end_db")}
            for s in ("before", "after")}
    print("mean", json.dumps(mean))
    if a.json:
        json.dump({"files": rows, "mean": mean, "before": a.before, "after": a.after}, open(a.json, "w", encoding="utf-8"),
                  indent=1)


if __name__ == "__main__":
    main()
