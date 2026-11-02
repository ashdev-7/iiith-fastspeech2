"""Kaggle job 05 (CPU): regenerate the English test outputs after the frontend fixes found in the
first full run (E3 "rhyth m" read as "R AY1 DH" + the letter M; E4 "andaroo" ending in OW).
No retraining: the checkpoint from fs2-03-train-en is reused, so only the text -> phones step differs.

Outputs: status.json, results_en_v2.tar (Results/English: test_generated_audios, variance_adaptor_plots,
variance_raw, frontend_trace.json, inference_report.json), eval_sentences_en.tar
(Evaluation/English_sentences: assignment + stress sentences with Whisper large-v3 transcripts, control
check), details.tar.
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
status = P.Status(job="reinfer_en", code_commit=commit, lang="en")
try:
    P.setup_python(status)
    P.run_tests(status, repo)
    with status.stage("stage_inputs") as rec:
        P.unpack_preprocessed("en", root)
        speaker = rec["test_voice"] = P.data_job_test_voice("en")
        ckpt = rec["checkpoint"] = P.find_checkpoint("en", "fs2-03-train-en")
    with status.stage("inference") as rec:
        results = f"{OUT}/Results/English"
        P.inference("en", root, repo, rec, ckpt, speaker, results, None)
        import json
        trace = json.load(open(f"{results}/frontend_trace.json", encoding="utf-8"))
        rec["rules"] = {k: v["rules"] for k, v in trace.items()}
        rec["key_words"] = {k: [(w["token"], w["source"], " ".join(w["phones"])) for w in v["words"]
                                if w["token"].lower() in ("rhythm", "rhyth", "m", "andaroo", "conduct")]
                            for k, v in trace.items()}
        P.tar(f"{OUT}/results_en_v2.tar", OUT, "Results")
        shutil.rmtree(f"{OUT}/Results")
    with status.stage("sentences_eval", fatal=False) as rec:
        out_dir = f"{OUT}/Evaluation/English_sentences"
        P.evaluate("en", root, repo, rec, ckpt, speaker, out_dir, ["--sentences_only"])
        P.tar(f"{OUT}/eval_sentences_en.tar", OUT, "Evaluation")
        shutil.rmtree(f"{OUT}/Evaluation")
finally:
    status.data["outputs"] = P.finalize()
    status.data["failed_stages"] = [k for k, v in status.data["stages"].items() if v.get("ok") is False]
    status.data["all_ok"] = not status.data["failed_stages"]
    status.write()
