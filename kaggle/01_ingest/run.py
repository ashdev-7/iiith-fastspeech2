"""Kaggle job 01: download both datasets, inventory, export candidate wavs, train/val split.

Outputs (/kaggle/working), kept to a handful of files so they can be fetched cheaply:
  raw_data_<lang>.tar                    <lang>/<speaker>/<utt>.wav|.lab (candidate clips)
  manifests/<lang>/inventory.tsv, train.tsv, val.tsv, not_selected.tsv,
                   split_summary.json, ingest_info.json, speakers_profile.json
  status.json, code_commit.txt
"""
import json
import os
import shutil
import subprocess
import sys
import traceback

OUT = "/kaggle/working"
TMP = "/kaggle/temp"
CACHE = f"{TMP}/hf"
REPO = "https://github.com/ashdev-7/iiith-fastspeech2.git"
REPO_REF = os.environ.get("FS2_REF", "main")  # pin a commit for exact reproduction
BALANCE = {"te": "equal", "en": "proportional"}  # plan.txt R2.3 / REVISION 4


def code_root():
    """Clone the project repo and record the exact commit used by this run."""
    dst = f"{TMP}/repo"
    subprocess.run(["git", "clone", "-q", REPO, dst], check=True)
    subprocess.run(["git", "-C", dst, "checkout", "-q", REPO_REF], check=True)
    commit = subprocess.run(["git", "-C", dst, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/code_commit.txt", "w") as f:
        f.write(commit + "\n")
    print("code commit:", commit, flush=True)
    return f"{dst}/fastspeech2", commit


def run(args, env):
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True, env=env)


def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unidecode", "inflect"], check=True)
    src, commit = code_root()
    status = {"code_commit": commit, "hf_token_secret": False, "languages": {}}
    env = dict(os.environ, PYTHONPATH=src, PYTHONIOENCODING="utf-8")
    try:
        from kaggle_secrets import UserSecretsClient
        env["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        status["hf_token_secret"] = True
    except Exception as e:
        status["hf_token_secret_error"] = f"{type(e).__name__}: {e}"
    print("HF_TOKEN secret available:", status["hf_token_secret"], flush=True)

    for lang in ("te", "en"):
        work = f"{TMP}/ingest"
        ingest = [sys.executable, f"{src}/data/ingest.py", "--lang", lang, "--cache", CACHE]
        try:
            run(ingest + ["--out", f"{TMP}/smoke", "--limit", "40"], env)  # smoke test first
            run(ingest + ["--out", work], env)
            run([sys.executable, f"{src}/data/select_split.py", "--manifest_dir", f"{work}/manifests/{lang}",
                 "--balance", BALANCE[lang]], env)
            shutil.copytree(f"{work}/manifests/{lang}", f"{OUT}/manifests/{lang}", dirs_exist_ok=True)
            run(["tar", "-cf", f"{OUT}/raw_data_{lang}.tar", "-C", f"{work}/raw_data", lang], env)
            shutil.rmtree(f"{work}/raw_data/{lang}")
            with open(f"{OUT}/manifests/{lang}/split_summary.json") as f:
                status["languages"][lang] = {"ok": True, "splits": json.load(f)["splits"]}
        except Exception as e:
            status["languages"][lang] = {"ok": False, "error": f"{type(e).__name__}: {e}",
                                         "traceback": traceback.format_exc()[-2000:]}
            print(f"!! {lang} failed: {e}", flush=True)
        shutil.rmtree(CACHE, ignore_errors=True)  # free /kaggle/temp before the next dataset
        with open(f"{OUT}/status.json", "w") as f:
            json.dump(status, f, indent=1)


if __name__ == "__main__":
    main()
