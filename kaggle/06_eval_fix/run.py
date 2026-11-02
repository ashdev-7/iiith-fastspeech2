"""Kaggle job 06 (GPU T4, ~20 min): re-runs the parts of fs2-04-eval that later fixes changed.
  * Telugu: full evaluate_val (its Whisper stage ran out of GPU memory in fs2-04-eval; now retries one
    clip at a time). Validation metrics are deterministic and identical to fs2-04-eval.
  * English: --sentences_only (frontend fixes of job 05, Hz-domain pitch/energy controls = M12, and
    Whisper's English normaliser for the CER of the sentences). The validation-set numbers of
    fs2-04-eval are unaffected by these changes and are kept.
Outputs: status.json, eval_te_v2.tar, eval_en_sentences_v2.tar, details.tar.
"""
import os
import shutil
import subprocess
import sys

REPO = "https://github.com/ashdev-7/iiith-fastspeech2.git"
REF = os.environ.get("FS2_REF", "main")
TMP, OUT = "/kaggle/temp", "/kaggle/working"
repo = f"{TMP}/repo"
subprocess.run(["git", "clone", "-q", REPO, repo], check=True)
subprocess.run(["git", "-C", repo, "checkout", "-q", REF], check=True)
commit = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
os.makedirs(OUT, exist_ok=True)
open(f"{OUT}/code_commit.txt", "w").write(commit + "\n")
sys.path.insert(0, f"{repo}/kaggle")
import pipeline as P  # noqa: E402

root = f"{TMP}/data"
status = P.Status(job="eval_fix", code_commit=commit)
try:
    P.setup_python(status)
    P.run_tests(status, repo)
    # Version 1 did both; its English part succeeded (eval_en_sentences_v2.tar). Version 2 repeats
    # Telugu only: its Whisper stage failed (batch OOM, then the pipeline had consumed its input dicts).
    for lang, name, extra in (("te", "Telugu", []),):
        with status.stage(f"{lang}_evaluate", fatal=False) as rec:
            P.unpack_preprocessed(lang, root)
            speaker = rec["speaker"] = P.data_job_test_voice(lang)
            ckpt = rec["checkpoint"] = P.find_checkpoint(lang, f"fs2-03-train-{lang}")
            out_dir = f"{OUT}/Evaluation/{name}"
            P.evaluate(lang, root, repo, rec, ckpt, speaker, out_dir, extra)
            tar_name = "eval_te_v2.tar" if lang == "te" else "eval_en_sentences_v2.tar"
            P.tar(f"{OUT}/{tar_name}", OUT, f"Evaluation/{name}")
            shutil.rmtree(out_dir)
finally:
    status.data["outputs"] = P.finalize()
    status.data["failed_stages"] = [k for k, v in status.data["stages"].items() if v.get("ok") is False]
    status.data["all_ok"] = not status.data["failed_stages"]
    status.write()
