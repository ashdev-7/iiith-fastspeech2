"""Stage "select": deterministic, duration-targeted train/val selection from inventory.tsv.

Rule (plan.txt 5.2, R2.3, R1.5):
  * units of assignment are groups: a Svarah recording (all of its chunks), or for Telugu all
    utterances with the same normalised transcript (both speakers often read the same
    sentence), so no audio or text crosses the split;
  * groups are visited in an order fixed by SHA-1(seed:group), i.e. a seeded pseudo-random
    order that does not depend on file order;
  * each speaker gets train/val quotas proportional to its share of candidate audio
    (train_hours, val_hours in total); a group goes to val if it fits every val quota it
    touches, else to train if it fits the train quotas, else it is left "not_selected";
  * leakage checks: utterance ids, groups, normalised transcripts and PCM hashes must be
    disjoint between train and val (duplicate audio within the inventory is excluded).
Limitation: the fill is greedy. A multi-speaker group (same sentence read by several speakers)
that no longer fits one speaker's quota is skipped, even if another speaker is still short.
This matters only when most groups are shared; the achieved hours per speaker are reported in
split_summary.json and must be checked.

Usage:
  python -m data.select_split --manifest_dir /kaggle/working/manifests/te --train_hours 5 --val_hours 1
"""
import argparse
import collections
import csv
import hashlib
import json
import os

FIELDS_OUT = ["utt_id", "speaker", "group", "duration", "model_text", "split"]


def read_inventory(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def select(rows, train_hours, val_hours, seed, tolerance=0.02, balance="proportional"):
    """balance="proportional": speaker quotas follow each speaker's share of candidate audio
    (used for Svarah's ~110 speakers); "equal": every speaker gets the same quota (used for the
    two IndicTTS Telugu speakers, whose usable audio differs a lot)."""
    cand = [r for r in rows if r["status"] == "candidate"]
    # Exclude exact duplicate audio (keep the first occurrence) before splitting.
    seen_hash, excluded = {}, {}
    for r in cand:
        if r["pcm_sha1"] in seen_hash:
            excluded[r["utt_id"]] = f"duplicate audio of {seen_hash[r['pcm_sha1']]}"
        else:
            seen_hash[r["pcm_sha1"]] = r["utt_id"]
    cand = [r for r in cand if r["utt_id"] not in excluded]
    for r in cand:
        r["duration"] = float(r["duration"])
        r["group"] = r["group"] or "text:" + hashlib.sha1(r["model_text"].encode("utf-8")).hexdigest()[:16]

    spk_total = collections.Counter()
    for r in cand:
        spk_total[r["speaker"]] += r["duration"]
    total = sum(spk_total.values())
    share = {s: (1 / len(spk_total) if balance == "equal" else spk_total[s] / total) for s in spk_total}
    quota = {split: {s: hours * 3600 * share[s] * (1 + tolerance) for s in spk_total}
             for split, hours in (("train", train_hours), ("val", val_hours))}
    short = {s: round(spk_total[s] / 3600, 3) for s in spk_total
             if spk_total[s] < (train_hours + val_hours) * 3600 * share[s]}
    if short:
        print(f"WARNING: speakers with less candidate audio than their quota (hours): {short}")
    used = {"train": collections.Counter(), "val": collections.Counter()}

    groups = collections.defaultdict(list)
    for r in cand:
        groups[r["group"]].append(r)
    order = sorted(groups, key=lambda g: hashlib.sha1(f"{seed}:{g}".encode("utf-8")).hexdigest())
    for g in order:
        need = collections.Counter()
        for r in groups[g]:
            need[r["speaker"]] += r["duration"]
        split = "not_selected"
        for candidate_split in ("val", "train"):
            if all(used[candidate_split][s] + d <= quota[candidate_split][s] for s, d in need.items()):
                split = candidate_split
                used[split].update(need)
                break
        for r in groups[g]:
            r["split"] = split
    return cand, excluded


def leakage_report(selected):
    by_split = {s: [r for r in selected if r["split"] == s] for s in ("train", "val")}
    keys = {"utt_id": "utt_id", "group": "group", "model_text": "model_text"}
    report = {}
    for name, key in keys.items():
        a = {r[key] for r in by_split["train"]}
        b = {r[key] for r in by_split["val"]}
        report[f"{name}_overlap"] = len(a & b)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_dir", required=True)
    ap.add_argument("--train_hours", type=float, default=5.0)
    ap.add_argument("--val_hours", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--balance", choices=["proportional", "equal"], default="proportional")
    ap.add_argument("--out_dir", default=None, help="where to write the split files (default: manifest_dir)")
    args = ap.parse_args()

    rows = read_inventory(os.path.join(args.manifest_dir, "inventory.tsv"))
    selected, excluded = select(rows, args.train_hours, args.val_hours, args.seed, balance=args.balance)
    out_dir = args.out_dir or args.manifest_dir
    os.makedirs(out_dir, exist_ok=True)
    leak = leakage_report(selected)
    if any(leak.values()):
        raise SystemExit(f"train/val overlap detected: {leak}")

    for split in ("train", "val", "not_selected"):
        with open(os.path.join(out_dir, f"{split}.tsv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS_OUT, delimiter="\t", lineterminator="\n", extrasaction="ignore")
            w.writeheader()
            w.writerows(sorted((r for r in selected if r["split"] == split), key=lambda r: r["utt_id"]))

    summary = {"seed": args.seed, "balance": args.balance, "targets_hours": {"train": args.train_hours, "val": args.val_hours},
               "inventory_rows": len(rows),
               "rejected": collections.Counter(r["reason"].split(":")[0] for r in rows if r["status"] == "rejected"),
               "duplicate_audio_excluded": len(excluded), "leakage": leak, "splits": {}}
    for split in ("train", "val", "not_selected"):
        part = [r for r in selected if r["split"] == split]
        per_spk = collections.Counter()
        for r in part:
            per_spk[r["speaker"]] += r["duration"]
        summary["splits"][split] = {
            "utterances": len(part), "hours": round(sum(r["duration"] for r in part) / 3600, 4),
            "speakers": len(per_spk),
            "hours_per_speaker_min_max": [round(min(per_spk.values(), default=0) / 3600, 4),
                                          round(max(per_spk.values(), default=0) / 3600, 4)]}
    with open(os.path.join(out_dir, "split_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
