"""Kaggle job 18 (CPU): is the Telugu validation evaluation affected by the aligner having seen the
validation clips? A second Telugu MFA aligner is trained on the TRAINING clips only (same dictionary,
same `mfa train` command as job 02), the validation clips are aligned with it (`mfa align`), and
tools/aligner_check.py compares the two alignments and the released model's duration scores against each.
Inputs: fs2-01-ingest (audio + manifests), fs2-02-data (preprocessed features + TextGrids), fs2-03-train-te.
Output: status.json (all numbers), aligner_check.json + per-utterance CSV, MFA logs in details.tar.
"""
import csv
import os
import shutil
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

lang, root = "te", f"{TMP}/data"
status = P.Status(job="aligner_check", code_commit=commit)
try:
    P.setup_python(status)
    P.run_tests(status, repo)
    P.install_mfa(status)
    with status.stage("inputs") as rec:
        P.unpack_ingest(lang, root)
        pre = P.unpack_preprocessed(lang, root)
        old = f"{TMP}/old_tg"
        os.makedirs(old, exist_ok=True)
        P.run(["tar", "-xf", P.find_one("/kaggle/input/**/TextGrid_te.tar", "TextGrid_te.tar"), "-C", old])
        ckpt = P.find_checkpoint(lang, "fs2-03-train-te")
        rec["checkpoint"] = ckpt
    with status.stage("corpora") as rec:
        corpus = f"{TMP}/corpus_te"
        d = f"{OUT}/mfa_te"
        os.makedirs(d, exist_ok=True)
        P.run([sys.executable, f"{repo}/fastspeech2/data/mfa_prepare.py", "--lang", lang, "--ingest_dir", root,
               "--corpus", corpus, "--dict_out", f"{d}/dictionary.txt"], env=P.code_env(repo))
        for split in ("train", "val"):  # split the corpus by the manifests: the aligner sees train only
            rows = list(csv.DictReader(open(f"{root}/manifests/{lang}/{split}.tsv", encoding="utf-8"), delimiter="\t"))
            n = 0
            for r in rows:
                for ext in (".wav", ".lab"):
                    src = f"{corpus}/{r['speaker']}/{r['utt_id']}{ext}"
                    if os.path.lexists(src):
                        dst = f"{TMP}/corpus_{split}/{r['speaker']}/{r['utt_id']}{ext}"
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        shutil.move(src, dst)
                        n += ext == ".wav"
            rec[split] = {"manifest_clips": len(rows), "corpus_clips": n}
        left = [os.path.join(a, f) for a, _, fs in os.walk(corpus) for f in fs]
        rec["unassigned_files"] = len(left)
        if left or rec["train"]["corpus_clips"] == 0 or rec["val"]["corpus_clips"] == 0:
            raise RuntimeError("corpus split incomplete")
    new_train, new_val = f"{TMP}/new_tg_train", f"{TMP}/new_tg_val"
    model_zip = f"{d}/telugu_acoustic_model_train_only.zip"
    with status.stage("mfa_train_on_train_only") as rec:
        P.mfa_writable(new_train, new_val, f"{TMP}/mfa_tmp_te", d)
        common = ["--clean", "-j", str(os.cpu_count() or 4), "--temporary_directory", f"{TMP}/mfa_tmp_te"]
        P.mfa("train", f"{TMP}/corpus_train", f"{d}/dictionary.txt", model_zip, "--output_directory", new_train,
              *common, log=f"{d}/commands.log")
        rec["model_bytes"] = os.path.getsize(model_zip)
    with status.stage("mfa_align_val") as rec:
        P.mfa("align", f"{TMP}/corpus_val", f"{d}/dictionary.txt", model_zip, new_val, *common, log=f"{d}/commands.log")
    P.save_mfa_logs(lang)
    with status.stage("compare") as rec:
        P.run([sys.executable, f"{repo}/tools/aligner_check.py", "--config", f"{repo}/Telugu/config.yaml",
               "--data_root", root, "--checkpoint", ckpt, "--old_tg", f"{old}/TextGrid", "--new_tg", new_val,
               "--new_train_tg", new_train, "--out", f"{OUT}/aligner_check.json"], env=P.code_env(repo))
        import json
        rec.update(json.load(open(f"{OUT}/aligner_check.json", encoding="utf-8")))
finally:
    status.data["outputs"] = P.finalize(keep=("status.json", "pytest.txt", "environment.txt", "code_commit.txt",
                                              "aligner_check.json", "aligner_check_per_utterance.csv"))
    status.data["failed_stages"] = [k for k, v in status.data["stages"].items() if v.get("ok") is False]
    status.data["all_ok"] = not status.data["failed_stages"]
    status.write()
