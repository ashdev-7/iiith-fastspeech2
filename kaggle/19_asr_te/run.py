"""Kaggle job 19 (GPU): Telugu intelligibility with a Telugu ASR model.
Whisper large-v3 is unreliable on Telugu (33 % CER even on copy-synthesis of real recordings; several test
transcripts above 100 % CER). This job re-measures with vasista22/whisper-telugu-large-v2 (pinned revision in
evaluate_val.ASR_REVISIONS; its training data does not include IndicTTS), same metric and same protocol:
  smoke : 5 validation clips, 3 ASR clips (fails fast if the ASR cannot load or decode)
  main  : full validation set (duration/pitch/energy metrics as before), ASR on 150 validation clips
          (synthesis vs copy-synthesis ceiling), assignment + extended sentences
  stress: assignment + stress sentences (sentences only)
The model and the Telugu Results are unchanged. Output: status.json (+ metrics in details.tar, no transcripts
of validation clips leave the job outside the tar).
"""
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

lang, root, voice = "te", f"{TMP}/data", "te_female"
ASR = ["--asr_model", "vasista22/whisper-telugu-large-v2"]
ext = f"{repo}/Evaluation/Extended/extended_sentences.json"
status = P.Status(job="asr_te", code_commit=commit)
try:
    P.setup_python(status)
    P.run_tests(status, repo)
    with status.stage("inputs"):
        P.unpack_preprocessed(lang, root)
        ckpt = P.find_checkpoint(lang, "fs2-03-train-te")
    with status.stage("smoke") as rec:
        m = P.evaluate(lang, root, repo, rec, ckpt, voice, f"{TMP}/smoke",
                       ASR + ["--max_utts", "5", "--asr_utts", "3", "--cpu_utts", "1", "--stress_file", ext])
        if "error" in (m.get("intelligibility") or {}):
            raise RuntimeError(m["intelligibility"]["error"])
    with status.stage("main") as rec:
        m = P.evaluate(lang, root, repo, rec, ckpt, voice, f"{OUT}/main", ASR + ["--stress_file", ext])
        rec["transcripts"] = {x["id"]: x.get("asr_transcript") for x in m["sentences"]}
    with status.stage("stress", fatal=False) as rec:
        m = P.evaluate(lang, root, repo, rec, ckpt, voice, f"{OUT}/stress", ASR + ["--sentences_only"])
        rec["transcripts"] = {x["id"]: x.get("asr_transcript") for x in m["sentences"]}
finally:
    for d in ("main", "stress"):  # validation transcripts stay out of the repo: drop them from the outputs
        p = f"{OUT}/{d}/asr_validation.csv"
        if os.path.exists(p):
            os.remove(p)
    status.data["outputs"] = P.finalize()
    status.data["failed_stages"] = [k for k, v in status.data["stages"].items() if v.get("ok") is False]
    status.data["all_ok"] = not status.data["failed_stages"]
    status.write()
