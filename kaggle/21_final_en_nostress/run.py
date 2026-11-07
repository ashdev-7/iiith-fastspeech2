"""Kaggle job 21 (CPU): job 17 without the lexical-stress rule (--prosody_stress 0), everything else equal.
Every measurement so far found the stress rule lowers intelligibility (Whisper CER: job 12 4.9 -> 6.7 %,
job 14 2.1 -> 3.2 %) and every variant with it garbles E4's tail; job 17 (with it) gave E4 17.1 % CER.
  fall + phrasing, duration 1.10, long words x1.25 with micro-pauses, GV post-filter 1.0,
  mel denoising (Results/English/noise_profile.json), edge fade.
Outputs: status.json, final_en.tar, details.tar.
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
ext = f"{repo}/Evaluation/Extended/extended_sentences.json"
merged = f"{TMP}/merged_sentences.json"
items = []
for f in (f"{repo}/Evaluation/stress_sentences.json", ext):
    items += json.load(open(f, encoding="utf-8"))["items"]
json.dump({"items": items}, open(merged, "w", encoding="utf-8"), ensure_ascii=False)
VOICE = "svarah_8a8e2655"
FINAL = ["--gv_ratio", f"{repo}/Results/English/gv_ratio.json", "--gv_strength", "1.0",
         "--prosody_fall", "--prosody_phrasing", "--duration_control", "1.10",
         "--long_word_scale", "1.25", "--heavy_word_pause",
         "--denoise_profile", f"{repo}/Results/English/noise_profile.json", "--edge_fade"]
base = f"{TMP}/final_en"
status = P.Status(job="final_en_nostress", code_commit=commit, lang="en")
try:
    P.setup_python(status)
    P.run_tests(status, repo)
    with status.stage("stage_inputs"):
        P.unpack_preprocessed("en", root)
        ckpt = P.find_checkpoint("en", "fs2-03-train-en")
    cfg = f"{repo}/{P.CONFIG['en']}"
    for tag, sent in (("Results", None), ("Extended", ext)):
        with status.stage(f"render_{tag}"):
            P.run([sys.executable, f"{repo}/fastspeech2/inference.py", "--config", cfg, "--data_root", root,
                   "--checkpoint", ckpt, "--speaker", VOICE, "--out_dir", f"{base}/{tag}"]
                  + (["--sentences", sent] if sent else []) + FINAL, env=P.code_env(repo))
    with status.stage("asr", fatal=False) as rec:
        m = P.evaluate("en", root, repo, rec, ckpt, VOICE, f"{base}/Evaluation",
                       ["--sentences_only", "--stress_file", merged] + FINAL)
        rec["transcripts"] = {x["id"]: x.get("asr_transcript") for x in m["sentences"]}
    P.tar(f"{OUT}/final_en.tar", TMP, "final_en", exclude=("*asr_validation.csv",))
finally:
    status.data["outputs"] = P.finalize()
    status.data["failed_stages"] = [k for k, v in status.data["stages"].items() if v.get("ok") is False]
    status.data["all_ok"] = not status.data["failed_stages"]
    status.write()
