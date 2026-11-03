"""Kaggle job 09 (GPU T4): HiFi-GAN fine-tuning on ground-truth-aligned FastSpeech 2 mels, for LANG.
Stages: tests -> unpack data + TextGrids -> smoke fine-tune (30 steps, catches bugs in minutes) ->
fine-tune under a watchdog (budget below) -> evaluation with the fine-tuned vocoder: test sentences
(Results layout), full validation evaluation incl. Whisper, natural/copy/oracle/free diagnosis.
Outputs: status.json, vocoder_<lang>.tar (generator_ft*.pth.tar, losses.csv, val.csv, gta_check.json),
results_<lang>_vft.tar, eval_<lang>_vft.tar, diag_<lang>_vft.json, diag_wavs_<lang>.tar, details.tar.
"""
import glob
import json
import os
import shutil
import subprocess
import sys

LANG = "te"
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
NAME = {"en": "English", "te": "Telugu"}[LANG]
VOICE = {"en": "svarah_8a8e2655", "te": "te_female"}[LANG]
BUDGET_MIN = {"en": 240, "te": 150}[LANG]
cfg = f"{repo}/{P.CONFIG[LANG]}"
status = P.Status(job=f"vocoder_ft_{LANG}", code_commit=commit, lang=LANG)
try:
    P.setup_python(status)
    with status.stage("pip_parselmouth"):
        P.run([sys.executable, "-m", "pip", "install", "-q", "praat-parselmouth"])
    P.run_tests(status, repo)
    with status.stage("stage_inputs") as rec:
        P.unpack_ingest(LANG, root)
        P.unpack_preprocessed(LANG, root)
        P.run(["tar", "-xf", P.find_one(f"/kaggle/input/**/TextGrid_{LANG}.tar", "TextGrids"),
               "-C", f"{root}/preprocessed/{LANG}"])
        rec["textgrids"] = len(glob.glob(f"{root}/preprocessed/{LANG}/TextGrid/*/*.TextGrid"))
        ckpt = rec["checkpoint"] = P.find_checkpoint(LANG, f"fs2-03-train-{LANG}")
    ft = [sys.executable, f"{repo}/fastspeech2/vocoder_finetune.py", "--config", cfg, "--data_root", root,
          "--checkpoint", ckpt]
    with status.stage("smoke_finetune") as rec:
        P.run(ft + ["--out_dir", f"{TMP}/vft_smoke", "--limit", "20", "--val_utts", "4", "--max_steps", "30",
                    "--d_warmup", "10", "--val_every", "15", "--save_every", "15"], env=P.code_env(repo))
        rec["gta_check"] = json.load(open(f"{TMP}/vft_smoke/gta_check.json"))
        rec["val"] = open(f"{TMP}/vft_smoke/val.csv").read()
    vdir = f"{OUT}/vocoder_{LANG}"
    with status.stage("finetune") as rec:
        killed = P.run_watched(ft + ["--out_dir", vdir, "--max_minutes", str(BUDGET_MIN)], P.code_env(repo),
                               f"{OUT}/finetune.log", BUDGET_MIN + 40, f"{vdir}/losses.csv", 20)
        rec["killed"] = killed
        rec["gta_check"] = json.load(open(f"{vdir}/gta_check.json"))
        rec["val"] = open(f"{vdir}/val.csv").read() if os.path.exists(f"{vdir}/val.csv") else None
        if not os.path.exists(f"{vdir}/generator_ft.pth.tar"):
            raise RuntimeError("no fine-tuned generator written")
    voc = f"{vdir}/generator_ft.pth.tar"
    font = P.fetch_telugu_font()
    with status.stage("test_sentences", fatal=False) as rec:
        res = f"{TMP}/res/Results/{NAME}"
        args = [sys.executable, f"{repo}/fastspeech2/inference.py", "--config", cfg, "--data_root", root,
                "--checkpoint", ckpt, "--speaker", VOICE, "--out_dir", res, "--vocoder", voc]
        P.run(args + (["--font_dir", font] if font else []), env=P.code_env(repo))
        P.tar(f"{OUT}/results_{LANG}_vft.tar", f"{TMP}/res", "Results")
    with status.stage("diagnose", fatal=False) as rec:
        d = f"{TMP}/diag"
        P.run([sys.executable, f"{repo}/fastspeech2/diagnose_quality.py", "--config", cfg, "--data_root", root,
               "--checkpoint", ckpt, "--speaker", VOICE, "--out_dir", d, "--n", "30", "--save_wavs", "6",
               "--vocoder", voc], env=P.code_env(repo))
        shutil.copy(f"{d}/diagnosis.json", f"{OUT}/diag_{LANG}_vft.json")
        rec["summary"] = json.load(open(f"{d}/diagnosis.json"))["summary"]
        P.tar(f"{OUT}/diag_wavs_{LANG}.tar", d, "wavs")
    with status.stage("evaluate", fatal=False) as rec:
        out_dir = f"{OUT}/Evaluation/{NAME}"
        P.evaluate(LANG, root, repo, rec, ckpt, VOICE, out_dir, ["--vocoder", voc])
        P.tar(f"{OUT}/eval_{LANG}_vft.tar", OUT, "Evaluation", exclude=("*asr_validation.csv",))
        shutil.rmtree(f"{OUT}/Evaluation")
    P.tar(f"{OUT}/vocoder_{LANG}.tar", OUT, f"vocoder_{LANG}")
    shutil.rmtree(vdir)
finally:
    status.data["outputs"] = P.finalize(keep=("status.json", "pytest.txt", "environment.txt", "code_commit.txt",
                                              f"diag_{LANG}_vft.json"))
    status.data["failed_stages"] = [k for k, v in status.data["stages"].items() if v.get("ok") is False]
    status.data["all_ok"] = not status.data["failed_stages"]
    status.write()
