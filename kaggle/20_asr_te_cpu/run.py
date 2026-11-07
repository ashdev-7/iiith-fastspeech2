"""Kaggle job 20 (CPU): Telugu intelligibility with the Telugu ASR model, sized for the 12-hour CPU limit.
Job 19 (same protocol, full validation set) cannot finish on CPU: its 5-clip smoke stage alone took 111 min.
Same ASR (vasista22/whisper-telugu-large-v2, pinned in evaluate_val.ASR_REVISIONS; not trained on IndicTTS):
  sentences : assignment (5) + extended (10) + stress (8) sentences, transcripts recorded
  validation: 30 evenly spaced validation clips, synthesis vs copy-synthesis ceiling (corpus CER)
The model and Results/Telugu are unchanged. Validation transcripts never leave the job.
"""
import json
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
status = P.Status(job="asr_te_cpu", code_commit=commit)
try:
    P.setup_python(status)
    P.run_tests(status, repo)
    with status.stage("inputs") as rec:
        P.unpack_preprocessed(lang, root)
        ckpt = P.find_checkpoint(lang, "fs2-03-train-te")
        items = []
        for f, s in (("Evaluation/Extended/extended_sentences.json", "extended"), ("Evaluation/stress_sentences.json", "stress")):
            items += [dict(it, source=s) for it in json.load(open(f"{repo}/{f}", encoding="utf-8"))["items"] if it["lang"] == lang]
        extra = f"{TMP}/te_extra_sentences.json"
        json.dump({"items": items}, open(extra, "w", encoding="utf-8"), ensure_ascii=False)
        rec["extra_sentences"] = len(items)
    with status.stage("sentences") as rec:
        m = P.evaluate(lang, root, repo, rec, ckpt, voice, f"{OUT}/sentences", ASR + ["--sentences_only", "--stress_file", extra])
        rec["transcripts"] = {x["id"]: x.get("asr_transcript") for x in m["sentences"]}
        rec["model_input"] = {x["id"]: x.get("model_input") for x in m["sentences"]}
    with status.stage("validation", fatal=False) as rec:
        empty = f"{TMP}/no_sentences.json"
        json.dump({"items": []}, open(empty, "w", encoding="utf-8"))
        P.evaluate(lang, root, repo, rec, ckpt, voice, f"{OUT}/validation",
                   ASR + ["--max_utts", "30", "--asr_utts", "30", "--cpu_utts", "1", "--save_samples", "0",
                          "--stress_file", empty])
finally:
    for d in ("sentences", "validation"):
        p = f"{OUT}/{d}/asr_validation.csv"
        if os.path.exists(p):
            os.remove(p)
    status.data["outputs"] = P.finalize()
    status.data["failed_stages"] = [k for k, v in status.data["stages"].items() if v.get("ok") is False]
    status.data["all_ok"] = not status.data["failed_stages"]
    status.write()
