"""Kaggle job 11 (CPU): English speaker adaptation of the decoder to the test voice (fastspeech2/adapt_speaker.py).
Uses only the test voice's clips in train.txt; its clips in val.txt pick the step and are scored.
  D  decoder side adapted (decoder, mel_linear, postnet, speaker embedding)
  V  as D plus the variance adaptor
For each: adaptation curves; GV ratio re-estimated on the voice's train clips and scored on its val clips at
post-filter strengths 0 / 0.5 / 1.0 (postfilter_eval.py); test sentences rendered in the Results layout at
strengths 0 and 1.0; Whisper on the test + stress sentences at both strengths.
Outputs: status.json, adapt_en.tar (everything except checkpoints and validation wavs), adapt_ckpt.tar
(the adapted checkpoints), details.tar.
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
base = f"{TMP}/adapt"
status = P.Status(job="adapt_en", code_commit=commit, lang="en")
try:
    P.setup_python(status)
    with status.stage("pip_parselmouth"):
        P.run([sys.executable, "-m", "pip", "install", "-q", "praat-parselmouth"])
    P.run_tests(status, repo)
    with status.stage("stage_inputs"):
        P.unpack_preprocessed("en", root)
        ckpt = P.find_checkpoint("en", "fs2-03-train-en")
    for name, parts in (("D", "decoder"), ("V", "decoder_var")):
        d = f"{base}/{name}"
        with status.stage(f"{name}_adapt", fatal=False) as rec:
            P.run([sys.executable, f"{repo}/fastspeech2/adapt_speaker.py", "--config", cfg, "--data_root", root,
                   "--checkpoint", ckpt, "--speaker", VOICE, "--out_dir", d, "--parts", parts], env=P.code_env(repo))
            s = json.load(open(f"{d}/adapt_summary.json"))
            rec["start"], rec["best"] = s["start"], s["best"]
        ack = f"{d}/adapted.pth.tar"
        if not os.path.exists(ack):
            continue
        with status.stage(f"{name}_postfilter", fatal=False) as rec:
            P.run([sys.executable, f"{repo}/fastspeech2/postfilter_eval.py", "--config", cfg, "--data_root", root,
                   "--checkpoint", ack, "--speaker", VOICE, "--out_dir", f"{d}/pf", "--save_wavs", "3"],
                  env=P.code_env(repo))
            rec["summary"] = json.load(open(f"{d}/pf/postfilter_eval.json"))["summary"]
        for st in ("0", "1.0"):
            extra = ["--gv_ratio", f"{d}/pf/gv_ratio.json", "--gv_strength", st] if st != "0" else []
            with status.stage(f"{name}_pf{st}_render", fatal=False):
                P.run([sys.executable, f"{repo}/fastspeech2/inference.py", "--config", cfg, "--data_root", root,
                       "--checkpoint", ack, "--speaker", VOICE, "--out_dir", f"{d}/pf{st}/Results"] + extra,
                      env=P.code_env(repo))
            with status.stage(f"{name}_pf{st}_sentences", fatal=False) as rec:
                m = P.evaluate("en", root, repo, rec, ack, VOICE, f"{d}/pf{st}/Evaluation", ["--sentences_only"] + extra)
                rec["transcripts"] = {x["id"]: x.get("asr_transcript") for x in m["sentences"]}
        P.run(["mkdir", "-p", f"{TMP}/ckpt/{name}", f"{TMP}/valwavs"])
        P.run(["mv", ack, f"{TMP}/ckpt/{name}/"])
        if os.path.isdir(f"{d}/pf/wavs"):
            P.run(["mv", f"{d}/pf/wavs", f"{TMP}/valwavs/{name}"])
    P.tar(f"{OUT}/adapt_en.tar", TMP, "adapt", exclude=("*asr_validation.csv",))
    if os.path.isdir(f"{TMP}/ckpt"):
        P.tar(f"{OUT}/adapt_ckpt.tar", TMP, "ckpt")
    if os.path.isdir(f"{TMP}/valwavs"):
        P.tar(f"{OUT}/adapt_valwavs.tar", TMP, "valwavs")
finally:
    status.data["outputs"] = P.finalize()
    status.data["failed_stages"] = [k for k, v in status.data["stages"].items() if v.get("ok") is False]
    status.data["all_ok"] = not status.data["failed_stages"]
    status.write()
