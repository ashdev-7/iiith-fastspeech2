"""Kaggle job 10 (CPU; the weekly GPU quota was used up): English, Whisper scores of the chosen configuration (A). Originally: compare the final configurations on the test and stress sentences.
  A  universal HiFi-GAN + GV post-filter (strength 1.0)   (chosen by listening, job 08)
  B  fine-tuned HiFi-GAN (fs2-09-vocoder-ft-en)
  C  fine-tuned HiFi-GAN + GV post-filter (strength 1.0)
Each: Whisper large-v3 on the test + stress sentences (evaluate_val.py --sentences_only) and the test
sentences rendered in the Results layout; for B/C also the variance/flux/Praat scores on held-out clips of
the test voice (postfilter_eval.py). Outputs: status.json, combos_en.tar, details.tar.
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, f"{repo}/kaggle")
import pipeline as P  # noqa: E402

root = f"{TMP}/data"
VOICE = "svarah_8a8e2655"
cfg = f"{repo}/{P.CONFIG['en']}"
ratio = f"{repo}/Results/English/gv_ratio.json"
status = P.Status(job="combos_en", code_commit=commit, lang="en")
try:
    P.setup_python(status)
    with status.stage("pip_parselmouth"):
        P.run([sys.executable, "-m", "pip", "install", "-q", "praat-parselmouth"])
    P.run_tests(status, repo)
    with status.stage("stage_inputs") as rec:
        P.unpack_preprocessed("en", root)
        ckpt = P.find_checkpoint("en", "fs2-03-train-en")
        P.run(["tar", "-xf", P.find_one("/kaggle/input/**/vocoder_en.tar", "vocoder_en.tar"), "-C", TMP])
        ft = rec["vocoder"] = f"{TMP}/vocoder_en/generator_ft.pth.tar"
        rec["finetune_val"] = open(f"{TMP}/vocoder_en/val.csv").read()
    # v1: only A. The fine-tuned vocoder (B, C) was rejected on fs2-09-vocoder-ft-en's own evaluation
    # (copy-synthesis CER 11.6 -> 12.3 %, HNR 17.1 -> 14.9 dB, E4 CER 2.9 -> 17.1 %).
    combos = {"A_universal_pf1": [None, 1.0]}
    base = f"{TMP}/combos"
    for name, (voc, st) in combos.items():
        extra = (["--vocoder", voc] if voc else []) + (["--gv_ratio", ratio, "--gv_strength", str(st)] if st else [])
        with status.stage(f"{name}_render", fatal=False):
            P.run([sys.executable, f"{repo}/fastspeech2/inference.py", "--config", cfg, "--data_root", root,
                   "--checkpoint", ckpt, "--speaker", VOICE, "--out_dir", f"{base}/{name}/Results"] + extra,
                  env=P.code_env(repo))
        with status.stage(f"{name}_sentences", fatal=False) as rec:
            m = P.evaluate("en", root, repo, rec, ckpt, VOICE, f"{base}/{name}/Evaluation", ["--sentences_only"] + extra)
            rec["transcripts"] = {s["id"]: s.get("asr_transcript") for s in m["sentences"]}
        if voc:
            with status.stage(f"{name}_scores", fatal=False) as rec:
                d = f"{base}/{name}/scores"
                P.run([sys.executable, f"{repo}/fastspeech2/postfilter_eval.py", "--config", cfg, "--data_root", root,
                       "--checkpoint", ckpt, "--speaker", VOICE, "--out_dir", d, "--vocoder", voc,
                       "--strengths", "0", "1.0", "--save_wavs", "0"], env=P.code_env(repo))
                rec["summary"] = json.load(open(f"{d}/postfilter_eval.json"))["summary"]
    P.tar(f"{OUT}/combos_en.tar", TMP, "combos", exclude=("*asr_validation.csv",))
finally:
    status.data["outputs"] = P.finalize()
    status.data["failed_stages"] = [k for k, v in status.data["stages"].items() if v.get("ok") is False]
    status.data["all_ok"] = not status.data["failed_stages"]
    status.write()
