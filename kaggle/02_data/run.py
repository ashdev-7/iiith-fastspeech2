"""Kaggle job 02 (CPU): full data preparation for both languages, via kaggle/pipeline.py:
MFA alignment -> feature extraction -> HiFi-GAN copy-synthesis check (Telugu test-voice choice).
Input: fs2-01-ingest output. Outputs: TextGrid_<lang>.tar, preprocessed_<lang>.tar (features,
train/val lists, stats, speakers), status.json, pytest.txt, environment.txt, code_commit.txt,
details.tar (dictionaries, MFA logs, copy-synthesis JSON and sample wavs).
"""
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

status = P.Status(job="data", code_commit=commit)
try:
    P.setup_python(status)
    P.run_tests(status, repo)
    P.install_mfa(status)
    for lang in ("te", "en"):
        root = f"{TMP}/data"
        with status.stage(f"{lang}_stage_inputs", fatal=False):
            P.unpack_ingest(lang, root)
        if not status.ok(f"{lang}_stage_inputs"):
            continue
        with status.stage(f"{lang}_align", fatal=False) as rec:
            P.align(lang, root, repo, rec)
        P.save_mfa_logs(lang)
        if not status.ok(f"{lang}_align"):
            continue
        P.tar(f"{OUT}/TextGrid_{lang}.tar", f"{root}/preprocessed/{lang}", "TextGrid")
        with status.stage(f"{lang}_preprocess", fatal=False) as rec:
            P.preprocess(lang, root, repo, rec)
        if not status.ok(f"{lang}_preprocess"):
            continue
        P.tar(f"{OUT}/preprocessed_{lang}.tar", f"{root}/preprocessed/{lang}", exclude=("./TextGrid",))
        with status.stage(f"{lang}_copy_synthesis", fatal=False) as rec:
            P.copy_synthesis(lang, root, repo, f"{OUT}/copy_synthesis_{lang}", "20" if lang == "te" else "3", rec)
        with status.stage(f"{lang}_test_voice", fatal=False) as rec:
            rec["speaker"] = P.test_voice(root, lang, f"{OUT}/copy_synthesis_{lang}/copy_synthesis.json")
        subprocess.run(["rm", "-rf", root])
finally:
    status.data["outputs"] = P.finalize()
    status.data["failed_stages"] = [k for k, v in status.data["stages"].items() if v.get("ok") is False]
    status.data["all_ok"] = not status.data["failed_stages"]
    status.write()
