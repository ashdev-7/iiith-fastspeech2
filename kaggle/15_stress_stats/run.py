"""Kaggle job 15 (CPU): re-run of job 12 stage 1 only (it failed writing its JSON). Job 12: English prosody — is lexical stress realised in the data and by the model, and do the
inference-time prosody rules (fastspeech2/prosody_rules.py) help?
  1. stress_stats.py: stress realisation on ground-truth train/val clips and on the model's val predictions.
  2. Test sentences rendered (Results layout) and Whisper-scored with the test and stress sentences, all with
     the chosen configuration (test voice, universal HiFi-GAN, GV post-filter 1.0), for
       R0     rules off
       F      terminal fall
       FS50   terminal fall + lexical stress at strength 0.5
       FS100  terminal fall + lexical stress at strength 1.0
Outputs: status.json, prosody_en.tar, details.tar.
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
VOICE = "svarah_8a8e2655"
cfg = f"{repo}/{P.CONFIG['en']}"
pf = ["--gv_ratio", f"{repo}/Results/English/gv_ratio.json", "--gv_strength", "1.0"]
VARIANTS = {"R0": [], "F": ["--prosody_fall"], "FS50": ["--prosody_fall", "--prosody_stress", "0.5"],
            "FS100": ["--prosody_fall", "--prosody_stress", "1.0"]}
base = f"{TMP}/prosody"
status = P.Status(job="stress_stats", code_commit=commit, lang="en")
try:
    P.setup_python(status)
    P.run_tests(status, repo)
    with status.stage("stage_inputs"):
        P.unpack_preprocessed("en", root)
        ckpt = P.find_checkpoint("en", "fs2-03-train-en")
    with status.stage("stress_stats", fatal=False) as rec:
        P.run([sys.executable, f"{repo}/fastspeech2/stress_stats.py", "--config", cfg, "--data_root", root,
               "--checkpoint", ckpt, "--speaker", VOICE, "--out_dir", f"{base}/stress_stats"], env=P.code_env(repo))
        r = json.load(open(f"{base}/stress_stats/stress_stats.json"))
        rec["skipped"] = r["skipped_clips"]
        rec["rates"] = {k: {x: v[x] for x in ("words", "rate", "chance_rate", "pitch_rate", "duration_rate", "energy_rate")}
                        for k, v in r["results"].items()}
    for name, extra in {}.items():  # renders: see job 12
        with status.stage(f"{name}_render", fatal=False):
            P.run([sys.executable, f"{repo}/fastspeech2/inference.py", "--config", cfg, "--data_root", root,
                   "--checkpoint", ckpt, "--speaker", VOICE, "--out_dir", f"{base}/{name}/Results"] + pf + extra,
                  env=P.code_env(repo))
        with status.stage(f"{name}_sentences", fatal=False) as rec:
            m = P.evaluate("en", root, repo, rec, ckpt, VOICE, f"{base}/{name}/Evaluation", ["--sentences_only"] + pf + extra)
            rec["transcripts"] = {x["id"]: x.get("asr_transcript") for x in m["sentences"]}
    P.tar(f"{OUT}/stress_stats.tar", TMP, "prosody")
finally:
    status.data["outputs"] = P.finalize()
    status.data["failed_stages"] = [k for k, v in status.data["stages"].items() if v.get("ok") is False]
    status.data["all_ok"] = not status.data["failed_stages"]
    status.write()
