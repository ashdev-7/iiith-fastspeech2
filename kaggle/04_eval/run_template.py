"""Kaggle job 04: objective evaluation of the trained models (fastspeech2/evaluate_val.py).
Generated from kaggle/04_eval/run_template.py by tools/make_train_jobs.py (only MODE differs).

MODE "full"  (GPU T4; inputs fs2-02-data, fs2-03-train-te, fs2-03-train-en): all validation
             utterances, Whisper large-v3 on 150 per language, CPU speed on 20.
MODE "smoke" (CPU; input fs2-02-data only): the same code on a 150-step model trained here, 20
             utterances and Whisper tiny, to find bugs before the GPU run. Its numbers mean nothing.
Outputs: status.json (headline numbers), eval_<lang>.tar (Evaluation/<Lang>/...), details.tar.
"""
import os
import shutil
import subprocess
import sys

MODE = "@MODE@"
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

LANG_DIR = {"te": "Telugu", "en": "English"}
root = f"{TMP}/data"
status = P.Status(job=f"eval_{MODE}", code_commit=commit)
try:
    P.setup_python(status)
    if MODE == "full" and not status.data["stages"]["environment"].get("cuda_matmul_ok"):
        raise SystemExit("no working GPU in this session")
    P.run_tests(status, repo)
    with status.stage("asr_import") as rec:  # transformers is preinstalled on Kaggle; fail early if not
        import transformers
        rec["transformers"] = transformers.__version__
    for lang in ("te", "en"):
        with status.stage(f"{lang}_evaluate", fatal=False) as rec:
            P.unpack_preprocessed(lang, root)
            speaker = rec["speaker"] = P.data_job_test_voice(lang)
            out_dir = f"{OUT}/Evaluation/{LANG_DIR[lang]}"
            if MODE == "full":
                ckpt = P.find_checkpoint(lang, f"fs2-03-train-{lang}")
                extra = []
            else:
                cfg = P.derived_config(repo, lang, "smoke", {"train": {
                    "step": {"total_step": 150, "log_step": 50, "val_step": 100000, "save_step": 100000},
                    "path": {"ckpt_path": f"smoke/{lang}/ckpt", "log_path": f"smoke/{lang}/log",
                             "result_path": f"smoke/{lang}/result"}}})
                ckpt, _ = P.train(lang, root, repo, {}, cfg, "20", restore="0")
                extra = ["--max_utts", "20", "--asr_utts", "6", "--cpu_utts", "3", "--save_samples", "2",
                         "--asr_model", "openai/whisper-tiny"]
            rec["checkpoint"] = ckpt
            P.evaluate(lang, root, repo, rec, ckpt, speaker, out_dir, extra)
            P.tar(f"{OUT}/eval_{lang}.tar", OUT, f"Evaluation/{LANG_DIR[lang]}")
            shutil.rmtree(out_dir)  # already in the tar; keeps details.tar small
finally:
    status.data["outputs"] = P.finalize()
    status.data["failed_stages"] = [k for k, v in status.data["stages"].items() if v.get("ok") is False]
    status.data["all_ok"] = not status.data["failed_stages"]
    status.write()
