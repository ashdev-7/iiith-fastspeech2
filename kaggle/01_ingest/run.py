"""Kaggle job 01 (CPU): download both datasets, inventory, export candidate wavs, train/val split.

Outputs (/kaggle/working): raw_data/{en,te}/<speaker>/<utt>.wav|.lab and
manifests/{en,te}/{inventory,train,val,not_selected}.tsv, split_summary.json, ingest_info.json.
"""
import os
import subprocess
import sys

OUT = "/kaggle/working"
CACHE = "/kaggle/temp/hf"
REPO = "https://github.com/ashdev-7/iiith-fastspeech2.git"
REPO_REF = os.environ.get("FS2_REF", "main")  # pin a commit for exact reproduction


def code_root():
    """Clone the project repo and record the exact commit used by this run."""
    dst = "/kaggle/temp/repo"
    subprocess.run(["git", "clone", "-q", REPO, dst], check=True)
    subprocess.run(["git", "-C", dst, "checkout", "-q", REPO_REF], check=True)
    commit = subprocess.run(["git", "-C", dst, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/code_commit.txt", "w") as f:
        f.write(commit + "\n")
    print("code commit:", commit, flush=True)
    return f"{dst}/fastspeech2"


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
