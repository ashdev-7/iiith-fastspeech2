"""Kaggle job 14 (CPU): the extended evaluation set (Evaluation/Extended/extended_sentences.json, 10 English +
10 Telugu sentences beyond the assignment), through both models in the final configuration, with and without
the prosody rules. Per variant: wavs, duration/pitch/energy plots, variance_raw, frontend_trace (inference.py
--sentences), and Whisper large-v3 transcripts with CER (evaluate_val.py --sentences_only --stress_file).
  English (svarah_8a8e2655, GV post-filter 1.0): R0 rules off, F terminal fall, FS50 fall + stress 0.5
  Telugu (te_female): R0 rules off, F terminal fall
Outputs: status.json, extended.tar, details.tar.
"""
import json
import os
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
sent = f"{repo}/Evaluation/Extended/extended_sentences.json"
CONF = {
    "en": ("svarah_8a8e2655", ["--gv_ratio", f"{repo}/Results/English/gv_ratio.json", "--gv_strength", "1.0"],
           {"R0": [], "F": ["--prosody_fall"], "FS50": ["--prosody_fall", "--prosody_stress", "0.5"]}),
    "te": ("te_female", [], {"R0": [], "F": ["--prosody_fall"]}),
}
base = f"{TMP}/extended"
status = P.Status(job="extended", code_commit=commit)
try:
    P.setup_python(status)
    P.run_tests(status, repo)
    font = P.fetch_telugu_font()
    for lang, (voice, pf, variants) in CONF.items():
        with status.stage(f"{lang}_inputs"):
            P.unpack_preprocessed(lang, root)
            ckpt = P.find_checkpoint(lang, f"fs2-03-train-{lang}")
        cfg = f"{repo}/{P.CONFIG[lang]}"
        for name, extra in variants.items():
            d = f"{base}/{lang}/{name}"
            with status.stage(f"{lang}_{name}_render", fatal=False):
                P.run([sys.executable, f"{repo}/fastspeech2/inference.py", "--config", cfg, "--data_root", root,
                       "--checkpoint", ckpt, "--speaker", voice, "--out_dir", f"{d}/Results", "--sentences", sent]
                      + pf + extra + (["--font_dir", font] if font and lang == "te" else []), env=P.code_env(repo))
            with status.stage(f"{lang}_{name}_asr", fatal=False) as rec:
                m = P.evaluate(lang, root, repo, rec, ckpt, voice, f"{d}/Evaluation",
                               ["--sentences_only", "--stress_file", sent] + pf + extra)
                rec["transcripts"] = {x["id"]: x.get("asr_transcript") for x in m["sentences"]}
    P.tar(f"{OUT}/extended.tar", TMP, "extended", exclude=("*asr_validation.csv",))
finally:
    status.data["outputs"] = P.finalize()
    status.data["failed_stages"] = [k for k, v in status.data["stages"].items() if v.get("ok") is False]
    status.data["all_ok"] = not status.data["failed_stages"]
    status.write()
