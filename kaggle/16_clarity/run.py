"""Kaggle job 16 (CPU): the clear-speech options (fastspeech2/clarity.py, prosody_rules.py rule 3, careful
pronunciations in the English frontend), Whisper large-v3 scored on the assignment test sentences, the stress
sentences and the extended set (one merged sentence file), per variant:
  English (svarah_8a8e2655, GV post-filter 1.0)
    FS100   terminal fall + lexical stress 1.0 (the previous choice)
    FSE     FS100 + mel denoising + edge fade
    CLEAR   FSE + phrasing + duration 1.10 + long words x1.25 + micro-pause before long words
  Telugu (te_female)
    R0, F, FE (F + edge fade)
Each variant also gets inference.py renders (Results layout) of the test sentences and of the extended set.
Outputs: status.json, clarity.tar, details.tar.
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

FS100 = ["--prosody_fall", "--prosody_stress", "1.0"]
POST = ["--denoise_profile", f"{repo}/Results/English/noise_profile.json", "--edge_fade"]
CLEAR = ["--prosody_phrasing", "--duration_control", "1.10", "--long_word_scale", "1.25", "--heavy_word_pause"]
CONF = {
    "en": ("svarah_8a8e2655", ["--gv_ratio", f"{repo}/Results/English/gv_ratio.json", "--gv_strength", "1.0"],
           {"FS100": FS100, "FSE": FS100 + POST, "CLEAR": FS100 + POST + CLEAR}),
    "te": ("te_female", [], {"R0": [], "F": ["--prosody_fall"], "FE": ["--prosody_fall", "--edge_fade"]}),
}
base = f"{TMP}/clarity"
status = P.Status(job="clarity", code_commit=commit)
try:
    P.setup_python(status)
    P.run_tests(status, repo)
    font = P.fetch_telugu_font()
    for lang, (voice, pf, variants) in CONF.items():
        with status.stage(f"{lang}_inputs"):
            P.unpack_preprocessed(lang, root)
            ckpt = P.find_checkpoint(lang, f"fs2-03-train-{lang}")
        cfg = f"{repo}/{P.CONFIG[lang]}"
        fnt = ["--font_dir", font] if font and lang == "te" else []
        for name, extra in variants.items():
            d = f"{base}/{lang}/{name}"
            for tag, sent in (("Results", None), ("Extended", ext)):
                with status.stage(f"{lang}_{name}_render_{tag}", fatal=False):
                    P.run([sys.executable, f"{repo}/fastspeech2/inference.py", "--config", cfg, "--data_root", root,
                           "--checkpoint", ckpt, "--speaker", voice, "--out_dir", f"{d}/{tag}"]
                          + (["--sentences", sent] if sent else []) + pf + extra + fnt, env=P.code_env(repo))
            with status.stage(f"{lang}_{name}_asr", fatal=False) as rec:
                m = P.evaluate(lang, root, repo, rec, ckpt, voice, f"{d}/Evaluation",
                               ["--sentences_only", "--stress_file", merged] + pf + extra)
                rec["transcripts"] = {x["id"]: x.get("asr_transcript") for x in m["sentences"]}
    P.tar(f"{OUT}/clarity.tar", TMP, "clarity", exclude=("*asr_validation.csv",))
finally:
    status.data["outputs"] = P.finalize()
    status.data["failed_stages"] = [k for k, v in status.data["stages"].items() if v.get("ok") is False]
    status.data["all_ok"] = not status.data["failed_stages"]
    status.write()
