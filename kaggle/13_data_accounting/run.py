"""Kaggle job 13 (CPU, no model): exact data accounting for the report.
Per language and split: utterances and hours SELECTED (manifests/<lang>/{train,val}.tsv, data/select_split.py)
versus USED (ids in preprocessed/<lang>/{train,val}.txt, i.e. after alignment and preprocessing), in raw
selected audio and in trimmed mel frames, plus train/val id and transcript overlap of the used sets.
Output: status.json with everything in it.
"""
import csv
import glob
import os
import subprocess
import sys

TMP, OUT = "/kaggle/temp", "/kaggle/working"
os.makedirs(OUT, exist_ok=True)
REPO = "https://github.com/ashdev-7/iiith-fastspeech2.git"
repo = f"{TMP}/repo"
subprocess.run(["git", "clone", "-q", REPO, repo], check=True)
commit = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
open(f"{OUT}/code_commit.txt", "w").write(commit + "\n")
sys.path.insert(0, f"{repo}/kaggle")
import pipeline as P  # noqa: E402
import numpy as np  # noqa: E402

root = f"{TMP}/data"
status = P.Status(job="data_accounting", code_commit=commit)
SR, HOP = 22050, 256
try:
    for lang in ("en", "te"):
        with status.stage(f"{lang}_accounting", fatal=False) as rec:
            pre = P.unpack_preprocessed(lang, root)
            man = os.path.dirname(P.find_one(f"/kaggle/input/**/manifests/{lang}/train.tsv", f"{lang} manifests"))
            used_text = {}
            for split in ("train", "val"):
                sel = {r["utt_id"]: r for r in csv.DictReader(open(f"{man}/{split}.tsv", encoding="utf-8"), delimiter="\t")}
                lines = [l.rstrip("\n").split("|") for l in open(f"{pre}/{split}.txt", encoding="utf-8") if l.strip()]
                used = [(b, s) for b, s, _, _ in lines]
                used_text[split] = {t for _, _, _, t in lines}
                missing = [b for b, _ in used if b not in sel]
                frames = 0
                for b, s in used:
                    f = f"{pre}/mel/{s}-mel-{b}.npy"
                    frames += np.load(f, mmap_mode="r").shape[0]
                rec[split] = {
                    "selected_utts": len(sel), "selected_hours": round(sum(float(r["duration"]) for r in sel.values()) / 3600, 4),
                    "used_utts": len(used), "used_ids_not_in_manifest": len(missing),
                    "used_hours_raw": round(sum(float(sel[b]["duration"]) for b, _ in used if b in sel) / 3600, 4),
                    "used_hours_trimmed_mel": round(frames * HOP / SR / 3600, 4),
                    "used_speakers": len({s for _, s in used}),
                    "dropped_ids": sorted(set(sel) - {b for b, _ in used})[:60],
                }
            ids = {split: {l.split("|")[0] for l in open(f"{pre}/{split}.txt", encoding="utf-8") if l.strip()} for split in ("train", "val")}
            rec["overlap_ids"] = len(ids["train"] & ids["val"])
            rec["overlap_transcripts"] = len(used_text["train"] & used_text["val"])
            P.run(["rm", "-rf", pre])
finally:
    status.data["outputs"] = P.finalize()
    status.data["failed_stages"] = [k for k, v in status.data["stages"].items() if v.get("ok") is False]
    status.data["all_ok"] = not status.data["failed_stages"]
    status.write()
