"""Kaggle job 02 (CPU): Montreal Forced Aligner on the selected train+val utterances.

  en: pretrained english_us_arpa acoustic model + reference lexicon (+ g2p_en for OOV words)
  te: `mfa train` on the Telugu corpus with the rule-based G2P dictionary
Outputs (/kaggle/working): TextGrid_<lang>.tar (<lang>/<speaker>/<utt>.TextGrid), mfa/<lang>/dictionary.txt,
dictionary_info.json, alignment_summary.json, MFA logs.
MFA is installed with micromamba (conda-forge) so the job does not depend on the image's Python.
"""
import glob
import json
import os
import shutil
import subprocess
import sys

OUT = "/kaggle/working"
TMP = "/kaggle/temp"
REPO = "https://github.com/ashdev-7/iiith-fastspeech2.git"
REPO_REF = os.environ.get("FS2_REF", "main")  # pin a commit for exact reproduction
MAMBA = f"{TMP}/bin/micromamba"
MFA_ENV = f"{TMP}/mfa_env"
JOBS = str(os.cpu_count() or 4)


def run(args, **kw):
    print("+", " ".join(args), flush=True)
    return subprocess.run(args, check=True, **kw)


def find_one(pattern, what):
    hits = glob.glob(pattern, recursive=True)
    if not hits:
        listing = subprocess.run(["find", "/kaggle/input", "-maxdepth", "3"], capture_output=True, text=True).stdout
        raise SystemExit(f"{what} not found ({pattern}). /kaggle/input:\n{listing}")
    return hits[0]


def install_mfa():
    os.makedirs(TMP, exist_ok=True)
    run(["bash", "-c", f"cd {TMP} && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba"])
    run([MAMBA, "create", "-y", "-q", "-p", MFA_ENV, "-c", "conda-forge", "montreal-forced-aligner"])
    run([MAMBA, "run", "-p", MFA_ENV, "mfa", "version"])


def mfa(*args):
    return run([MAMBA, "run", "-p", MFA_ENV, "mfa", *args])


def summarise(lang, corpus, tg_dir):
    expected = {os.path.splitext(os.path.relpath(p, corpus))[0] for p in glob.glob(f"{corpus}/*/*.lab")}
    got = {os.path.splitext(os.path.relpath(p, tg_dir))[0] for p in glob.glob(f"{tg_dir}/*/*.TextGrid")}
    summary = {"lang": lang, "expected": len(expected), "aligned": len(got & expected),
               "missing": sorted(expected - got)[:200], "missing_count": len(expected - got)}
    with open(f"{OUT}/mfa/{lang}/alignment_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "missing"}), flush=True)


def clone_code():
    """Clone the project repo and record the exact commit used by this run."""
    dst = f"{TMP}/repo"
    run(["git", "clone", "-q", REPO, dst])
    run(["git", "-C", dst, "checkout", "-q", REPO_REF])
    commit = subprocess.run(["git", "-C", dst, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    with open(f"{OUT}/code_commit.txt", "w") as f:
        f.write(commit + "\n")
    print("code commit:", commit, flush=True)
    return f"{dst}/fastspeech2"


def unpack_ingest(lang):
    """Rebuild <TMP>/ingest/{raw_data,manifests}/<lang> from the fs2-01-ingest output files."""
    ingest = f"{TMP}/ingest"
    tar = find_one(f"/kaggle/input/**/raw_data_{lang}.tar", f"fs2-01-ingest raw_data_{lang}.tar")
    os.makedirs(f"{ingest}/raw_data", exist_ok=True)
    run(["tar", "-xf", tar, "-C", f"{ingest}/raw_data"])
    manifests = os.path.dirname(find_one(f"/kaggle/input/**/manifests/{lang}/train.tsv", f"{lang} manifests"))
    shutil.copytree(manifests, f"{ingest}/manifests/{lang}", dirs_exist_ok=True)
    return ingest


def main():
    src = clone_code()
    run([sys.executable, "-m", "pip", "install", "-q", "g2p_en", "unidecode", "inflect"])
    install_mfa()
    env = dict(os.environ, PYTHONPATH=src, PYTHONIOENCODING="utf-8")

    for lang in ("te", "en"):
        corpus, dict_path = f"{TMP}/corpus_{lang}", f"{OUT}/mfa/{lang}/dictionary.txt"
        tg_dir = f"{TMP}/TextGrid/{lang}"
        try:
            ingest = unpack_ingest(lang)
            run([sys.executable, f"{src}/data/mfa_prepare.py", "--lang", lang, "--ingest_dir", ingest,
                 "--corpus", corpus, "--dict_out", dict_path], env=env)
            common = ["--clean", "--overwrite", "-j", JOBS, "--temporary_directory", f"{TMP}/mfa_tmp_{lang}"]
            if lang == "en":
                mfa("model", "download", "acoustic", "english_us_arpa")
                mfa("align", corpus, dict_path, "english_us_arpa", tg_dir, "--beam", "100",
                    "--retry_beam", "400", *common)
            else:
                mfa("train", corpus, dict_path, f"{OUT}/mfa/te/telugu_acoustic_model.zip",
                    "--output_directory", tg_dir, *common)
            summarise(lang, corpus, tg_dir)
            run(["tar", "-cf", f"{OUT}/TextGrid_{lang}.tar", "-C", f"{TMP}/TextGrid", lang])
        except (subprocess.CalledProcessError, SystemExit) as e:
            print(f"!! {lang} failed: {e}", flush=True)
        finally:
            for log in glob.glob(f"{TMP}/mfa_tmp_{lang}/**/*.log", recursive=True)[:50]:
                dst = f"{OUT}/mfa/{lang}/logs/{os.path.relpath(log, f'{TMP}/mfa_tmp_{lang}')}"
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(log, dst)


if __name__ == "__main__":
    main()
