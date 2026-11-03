"""Kaggle job 08 (CPU, no GPU quota): GV post-filter against decoder over-smoothing.
Per language: estimate the per-bin variance ratio on training clips of the test voice, score strengths
0 / 0.5 / 1.0 on held-out validation clips (variance, flux, MCD, Praat), and render the test sentences at
strengths 0.5 and 1.0 (our synthesis only). Outputs: status.json, pf_<lang>.tar (gv_ratio.json,
postfilter_eval.json, Results-style test outputs per strength), pf_wavs.tar (validation A/B, not committed).
"""
import json
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
VOICE = {"en": "svarah_8a8e2655", "te": "te_female"}
status = P.Status(job="postfilter", code_commit=commit)
try:
    P.setup_python(status)
    with status.stage("pip_parselmouth"):
        P.run([sys.executable, "-m", "pip", "install", "-q", "praat-parselmouth"])
    P.run_tests(status, repo)
    font = P.fetch_telugu_font()
    for lang in ("en", "te"):
        with status.stage(f"{lang}_postfilter", fatal=False) as rec:
            P.unpack_preprocessed(lang, root)
            ckpt = P.find_checkpoint(lang, f"fs2-03-train-{lang}")
            out = f"{TMP}/pf_{lang}"
            P.run([sys.executable, f"{repo}/fastspeech2/postfilter_eval.py", "--config", f"{repo}/{P.CONFIG[lang]}",
                   "--data_root", root, "--checkpoint", ckpt, "--speaker", VOICE[lang], "--out_dir", out],
                  env=P.code_env(repo))
            rec["summary"] = json.load(open(f"{out}/postfilter_eval.json"))["summary"]
            rec["ratio_median"] = sorted(json.load(open(f"{out}/gv_ratio.json"))["ratio"])[40]
            for st in ("0.5", "1.0"):
                with status.stage(f"{lang}_render_gv{st}", fatal=False):
                    P.run([sys.executable, f"{repo}/fastspeech2/inference.py", "--config", f"{repo}/{P.CONFIG[lang]}",
                           "--data_root", root, "--checkpoint", ckpt, "--speaker", VOICE[lang],
                           "--out_dir", f"{out}/render_gv{st}", "--gv_ratio", f"{out}/gv_ratio.json",
                           "--gv_strength", st] + (["--font_dir", font] if font else []), env=P.code_env(repo))
            os.makedirs(f"{TMP}/pf_wavs", exist_ok=True)
            shutil.move(f"{out}/wavs", f"{TMP}/pf_wavs/{lang}")
            P.tar(f"{OUT}/pf_{lang}.tar", TMP, f"pf_{lang}")
    if os.path.isdir(f"{TMP}/pf_wavs"):
        P.tar(f"{OUT}/pf_wavs.tar", TMP, "pf_wavs")
finally:
    status.data["outputs"] = P.finalize()
    status.data["failed_stages"] = [k for k, v in status.data["stages"].items() if v.get("ok") is False]
    status.data["all_ok"] = not status.data["failed_stages"]
    status.write()
