"""Does the Telugu aligner having seen the validation clips change the validation results? (report 2.2)

The Telugu MFA aligner used for training was trained on the selected train + val clips (there is no
pretrained Telugu aligner). This check trains a second aligner on the TRAINING clips only, aligns the
validation clips with it (kaggle/18_aligner_check), and compares:

  A. the two alignments of the validation clips (and, for reference, of the training clips):
     same phone sequence?  per-phone duration difference (frames of 11.6 ms), phone-boundary shift (ms);
  B. the trained Telugu model's duration predictions (model unchanged) scored against each alignment,
     with the metric of evaluate_val.py (per-utterance MAE in frames and Pearson r of log(1+d) over
     non-silence phones, averaged over utterances).

Durations are taken from the TextGrids with the reference preprocessor's own get_alignment (leading and
trailing silences trimmed), so they are exactly what preprocessing would have produced.

  python tools/aligner_check.py --config Telugu/config.yaml --data_root D --checkpoint te_model.pth.tar \
      --old_tg D/preprocessed/te/TextGrid --new_tg NEW/TextGrid --new_train_tg NEW_TRAIN/TextGrid --out check.json
"""
import argparse
import glob
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import tgt
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fastspeech2"))

import eval_metrics as M  # noqa: E402
import inference as inf  # noqa: E402
from preprocessor.preprocessor import Preprocessor  # noqa: E402
from text import text_to_sequence  # noqa: E402
from utils.config import load_configs  # noqa: E402

SILENCES = {"sp", "spn", "sil"}


def read_alignment(path, sr, hop):
    """(phones, durations in frames, non-silence phone start times in s) as the preprocessor computes them."""
    tier = tgt.io.read_textgrid(path, include_empty_intervals=True).get_tier_by_name("phones")
    phones, durs, _, _ = Preprocessor.get_alignment(SimpleNamespace(sampling_rate=sr, hop_length=hop), tier)
    starts = [t.start_time for t in tier._objects if t.text not in ("", *SILENCES)]
    return phones, durs, starts


def compare(old_dir, new_dir, ids, sr, hop):
    """Agreement of two alignments of the same clips."""
    n_both = n_same = 0
    dd, shift, missing = [], [], []
    for spk, utt in ids:
        a, b = f"{old_dir}/{spk}/{utt}.TextGrid", f"{new_dir}/{spk}/{utt}.TextGrid"
        if not (os.path.exists(a) and os.path.exists(b)):
            missing.append(utt)
            continue
        n_both += 1
        pa, da, sa = read_alignment(a, sr, hop)
        pb, db, sb = read_alignment(b, sr, hop)
        if pa != pb:
            continue
        n_same += 1
        dd += [abs(x - y) for p, x, y in zip(pa, da, db) if p not in SILENCES]
        shift += [abs(x - y) * 1000 for x, y in zip(sa, sb)]
    dd, shift = np.array(dd, float), np.array(shift, float)
    return {"clips": len(ids), "aligned_by_both": n_both, "not_aligned_by_one_or_both": len(missing),
            "missing_examples": missing[:20], "same_phone_sequence": n_same,
            "phones_compared": int(len(dd)),
            "duration_abs_diff_frames": {"mean": float(dd.mean()), "median": float(np.median(dd)),
                                         "p95": float(np.percentile(dd, 95))} if len(dd) else None,
            "phones_identical_duration_pct": float(100 * np.mean(dd == 0)) if len(dd) else None,
            "phones_within_1_frame_pct": float(100 * np.mean(dd <= 1)) if len(dd) else None,
            "phones_within_2_frames_pct": float(100 * np.mean(dd <= 2)) if len(dd) else None,
            "boundary_shift_ms": {"mean": float(shift.mean()), "median": float(np.median(shift)),
                                  "p95": float(np.percentile(shift, 95))} if len(shift) else None}


def main():
    ap = argparse.ArgumentParser()
    for a in ("--config", "--data_root", "--checkpoint", "--old_tg", "--new_tg", "--out"):
        ap.add_argument(a, required=True)
    ap.add_argument("--old_train_tg", default=None, help="defaults to --old_tg")
    ap.add_argument("--new_train_tg", default=None, help="training-clip TextGrids of the new aligner")
    args = ap.parse_args()
    configs = load_configs(args.config, args.data_root)
    pre_cfg = configs[0]
    lang = pre_cfg["preprocessing"]["text"]["language"]
    sr, hop = pre_cfg["preprocessing"]["audio"]["sampling_rate"], pre_cfg["preprocessing"]["stft"]["hop_length"]
    pre = pre_cfg["path"]["preprocessed_path"]
    speakers = json.load(open(f"{pre}/speakers.json", encoding="utf-8"))

    def split_ids(name):
        return [tuple(l.split("|")[:2])[::-1] for l in open(f"{pre}/{name}.txt", encoding="utf-8") if l.strip()]
    val_ids = split_ids("val")  # (speaker, utt) of the validation clips used in the reported evaluation
    report = {"what": __doc__.strip().splitlines()[0], "language": lang,
              "A_val_alignment_agreement": compare(args.old_tg, args.new_tg, val_ids, sr, hop)}
    if args.new_train_tg:
        report["A_train_alignment_agreement"] = compare(args.old_train_tg or args.old_tg, args.new_train_tg,
                                                        split_ids("train"), sr, hop)

    model, step = inf.load_model(configs, args.checkpoint)
    report["checkpoint_step"] = step

    def predict(spk, phones):
        seq = text_to_sequence("{" + " ".join(phones) + "}", [], lang)
        with torch.no_grad():
            o = model(torch.tensor([speakers[spk]]).to(inf.device), torch.tensor([seq]).to(inf.device),
                      torch.tensor([len(seq)]).to(inf.device), len(seq))
        return o[5][0, :len(seq)].cpu().numpy().astype(float)

    def scores(phones, gt, pred):
        keep = np.array([p not in SILENCES for p in phones])
        return float(np.mean(np.abs(pred[keep] - gt[keep]))), M.pearson(np.log1p(pred[keep]), np.log1p(gt[keep]))

    rows = []
    for spk, utt in val_ids:
        # Reported targets: the stored duration features (what evaluate_val.py scored against).
        old_ph = next(l.split("|")[2] for l in open(f"{pre}/val.txt", encoding="utf-8") if l.startswith(utt + "|"))
        old_ph = old_ph.strip("{}").split()
        old_d = np.load(f"{pre}/duration/{spk}-duration-{utt}.npy").astype(float)
        row = {"utt": utt, "speaker": spk}
        row["old_mae"], row["old_log_r"] = scores(old_ph, old_d, predict(spk, old_ph))
        new = f"{args.new_tg}/{spk}/{utt}.TextGrid"
        if os.path.exists(new):
            new_ph, new_d, _ = read_alignment(new, sr, hop)
            row["new_mae"], row["new_log_r"] = scores(new_ph, np.array(new_d, float), predict(spk, new_ph))
            row["same_phones"] = new_ph == old_ph
        rows.append(row)

    both = [r for r in rows if "new_mae" in r]

    def summ(rs, k):
        s = M.summarise([r.get(k) for r in rs])
        return {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in s.items()}
    report["B_model_duration_scores"] = {
        "reported_targets_all_val": {"utterances": len(rows), "mae_frames": summ(rows, "old_mae"),
                                     "log_r": summ(rows, "old_log_r")},
        "same_clips_reported_targets": {"utterances": len(both), "mae_frames": summ(both, "old_mae"),
                                        "log_r": summ(both, "old_log_r")},
        "same_clips_train_only_aligner": {"utterances": len(both), "mae_frames": summ(both, "new_mae"),
                                          "log_r": summ(both, "new_log_r")},
        "note": "same metric as evaluate_val.py; the model is the released te_model, unchanged"}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(report, open(args.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    with open(os.path.splitext(args.out)[0] + "_per_utterance.csv", "w", encoding="utf-8") as f:
        keys = ["utt", "speaker", "old_mae", "old_log_r", "new_mae", "new_log_r", "same_phones"]
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(k) is None else str(r.get(k)) for k in keys) + "\n")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
