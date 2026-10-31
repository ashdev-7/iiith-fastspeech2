"""Kaggle job 01 (CPU): download both datasets, inventory, export candidate wavs, train/val split.

Outputs (/kaggle/working): raw_data/{en,te}/<speaker>/<utt>.wav|.lab and
manifests/{en,te}/{inventory,train,val,not_selected}.tsv, split_summary.json, ingest_info.json.
"""
import glob
import os
import shutil
import subprocess
import sys

OUT = "/kaggle/working"
CACHE = "/kaggle/temp/hf"


def code_root():
    """Locate the fs2-code dataset (mounted as a folder, or as project.zip) under /kaggle/input."""
    marker = "fastspeech2/data/ingest.py"
    hits = glob.glob(f"/kaggle/input/**/{marker}", recursive=True)
    if not hits:
        zips = glob.glob("/kaggle/input/**/project.zip", recursive=True)
        if zips:
            shutil.unpack_archive(zips[0], "/kaggle/temp/code")
            hits = glob.glob(f"/kaggle/temp/code/**/{marker}", recursive=True)
    hits = [h[: -len(marker)] + "fastspeech2" for h in hits]
    if not hits:
        listing = subprocess.run(["find", "/kaggle/input", "-maxdepth", "4"], capture_output=True, text=True).stdout
        raise SystemExit(f"fs2-code not found under /kaggle/input. Contents:\n{listing}")
    print("code root:", hits[0], flush=True)
    return hits[0]


def run(args, env):
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True, env=env)


def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unidecode", "inflect"], check=True)
    src = code_root()
    env = dict(os.environ, PYTHONPATH=src, PYTHONIOENCODING="utf-8")
    try:
        from kaggle_secrets import UserSecretsClient
        env["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        print("HF_TOKEN secret: found", flush=True)
    except Exception as e:
        print(f"HF_TOKEN secret: NOT available ({type(e).__name__}); the gated Svarah download will fail",
              flush=True)

    for lang in ("te", "en"):
        ingest = [sys.executable, os.path.join(src, "data", "ingest.py"), "--lang", lang, "--cache", CACHE]
        try:
            run(ingest + ["--out", "/kaggle/temp/smoke", "--limit", "40"], env)  # smoke test first
            run(ingest + ["--out", OUT], env)
            run([sys.executable, os.path.join(src, "data", "select_split.py"),
                 "--manifest_dir", os.path.join(OUT, "manifests", lang)], env)
        except subprocess.CalledProcessError as e:
            print(f"!! {lang} failed: {e}", flush=True)


if __name__ == "__main__":
    main()
