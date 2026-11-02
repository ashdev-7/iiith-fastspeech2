"""Kaggle job 07 (CPU, ~20 min, no GPU quota): where is the quality lost?
For 30 validation utterances of each test voice: natural recording vs copy-synthesis (vocoder only)
vs FastSpeech 2 with ground-truth variances (oracle) vs free-running FastSpeech 2, measured with Praat
(fastspeech2/diagnose_quality.py). Outputs: status.json, diag_en.json, diag_te.json (small, committed
to Evaluation/), diag_wavs.tar (a few A/B wavs for listening; dataset-derived, NOT committed).
"""
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
status = P.Status(job="diagnose", code_commit=commit)
try:
    P.setup_python(status)
    with status.stage("pip_parselmouth"):
        P.run([sys.executable, "-m", "pip", "install", "-q", "praat-parselmouth"])
    P.run_tests(status, repo)
    for lang in ("en", "te"):
        with status.stage(f"{lang}_diagnose", fatal=False) as rec:
            P.unpack_ingest(lang, root)
            P.unpack_preprocessed(lang, root)
            speaker = rec["speaker"] = P.data_job_test_voice(lang)
            ckpt = P.find_checkpoint(lang, f"fs2-03-train-{lang}")
            out_dir = f"{TMP}/diag_{lang}"
            P.run([sys.executable, f"{repo}/fastspeech2/diagnose_quality.py", "--config", f"{repo}/{P.CONFIG[lang]}",
                   "--data_root", root, "--checkpoint", ckpt, "--speaker", speaker, "--out_dir", out_dir,
                   "--n", "30", "--save_wavs", "6", "--rank_speakers"], env=P.code_env(repo))
            shutil.copy(f"{out_dir}/diagnosis.json", f"{OUT}/diag_{lang}.json")
            shutil.copy(f"{out_dir}/speaker_clarity.json", f"{OUT}/clarity_{lang}.json")
            import json
            ranked = json.load(open(f"{out_dir}/speaker_clarity.json"))["ranked"]
            rec["clarity_top"] = ranked[:5]
            if lang == "en":  # the test sentences in the clearest voices (our synthesis only: committable)
                for spk in ranked[:4]:
                    with status.stage(f"en_voice_{spk}", fatal=False):
                        P.run([sys.executable, f"{repo}/fastspeech2/inference.py", "--config", f"{repo}/{P.CONFIG[lang]}",
                               "--data_root", root, "--checkpoint", ckpt, "--speaker", spk,
                               "--out_dir", f"{TMP}/voices/{spk}"], env=P.code_env(repo))
            import json
            rec["summary"] = json.load(open(f"{out_dir}/diagnosis.json"))["summary"]
            os.makedirs(f"{TMP}/diag_wavs", exist_ok=True)
            shutil.copytree(f"{out_dir}/wavs", f"{TMP}/diag_wavs/{lang}", dirs_exist_ok=True)
    if os.path.isdir(f"{TMP}/voices"):
        P.tar(f"{OUT}/en_voices.tar", TMP, "voices")
    if os.path.isdir(f"{TMP}/diag_wavs"):
        P.tar(f"{OUT}/diag_wavs.tar", TMP, "diag_wavs")
finally:
    status.data["outputs"] = P.finalize(keep=("status.json", "pytest.txt", "environment.txt", "code_commit.txt",
                                              "diag_en.json", "diag_te.json", "clarity_en.json", "clarity_te.json"))
    status.data["failed_stages"] = [k for k, v in status.data["stages"].items() if v.get("ok") is False]
    status.data["all_ok"] = not status.data["failed_stages"]
    status.write()
