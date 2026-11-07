"""Is lexical stress realised in the English DATA, and by the model? (report §6)

For every clip: words from the project frontend on the clip's transcript (clips whose frontend phones differ
from the aligned phones in train/val.txt are skipped), then prosody_rules.stress_realised on
  - the ground-truth phone-level pitch / energy / duration (train and val), and
  - the model's free-running predictions (val).
Writes <out>/stress_stats.json, overall and for the test voice.
"""
import argparse
import json
import os

import numpy as np
import torch

import prosody_rules as R
from dataset import Dataset
from inference import device, frontend, load_model
from utils.config import load_configs


def tally(acc, key, res):
    a = acc.setdefault(key, {"clips": 0, "words": 0, "realised": 0, "pitch": 0, "duration": 0, "energy": 0, "chance": 0.0})
    n, ok, cues = res
    a["clips"] += 1
    a["words"] += int(n)
    a["realised"] += int(ok)
    for k, v in cues.items():
        a[k] += float(v) if k == "chance" else int(v)


def main(a):
    configs = load_configs(a.config, a.data_root)
    pre_cfg, _, train_cfg = configs
    model, step = load_model(configs, a.checkpoint)
    acc, skipped = {}, {"train": 0, "val": 0}
    for split in ("train", "val"):
        ds = Dataset(split + ".txt", pre_cfg, train_cfg, sort=False, drop_last=False)
        for i in range(len(ds)):
            phones = ds.text[i].strip("{}").split()
            try:
                fp, trace = frontend(ds.raw_text[i], "en")
                if [p for p in fp if p not in R.SILENCES] != [p for p in phones if p not in R.SILENCES]:
                    raise ValueError("frontend/alignment phones differ")
                s = ds[i]
                gt = R.stress_realised(phones, trace["words"], s["pitch"], s["energy"], s["duration"])
            except Exception:  # noqa: BLE001
                skipped[split] += 1
                continue
            voice = "voice" if ds.speaker[i] == a.speaker else "others"
            for key in (f"{split}_gt_all", f"{split}_gt_{voice}"):
                tally(acc, key, gt)
            if split == "val":
                with torch.no_grad():
                    out = model(torch.tensor([s["speaker"]], device=device).long(),
                                torch.from_numpy(np.array([s["text"]])).long().to(device),
                                torch.tensor([len(s["text"])], device=device), len(s["text"]))
                n = len(phones)
                pr = R.stress_realised(phones, trace["words"], out[2][0, :n].cpu().numpy(),
                                       out[3][0, :n].cpu().numpy(), out[5][0, :n].cpu().numpy())
                for key in ("val_pred_all", f"val_pred_{voice}"):
                    tally(acc, key, pr)
        print(split, "done", flush=True)
    for v in acc.values():
        v["rate"] = round(v["realised"] / max(v["words"], 1), 3)
        v["chance_rate"] = round(v["chance"] / max(v["words"], 1), 3)
        for k in ("pitch", "duration", "energy"):
            v[k + "_rate"] = round(v[k] / max(v["words"], 1), 3)
    res = {"checkpoint_step": step, "speaker": a.speaker, "skipped_clips": skipped, "results": acc,
           "definition": R.stress_realised.__doc__}
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(a.out_dir, "stress_stats.json"), "w"), indent=1, default=lambda o: o.item())
    print(json.dumps(res, indent=1, default=lambda o: o.item()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--out_dir", required=True)
    main(ap.parse_args())
