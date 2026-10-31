"""Kaggle job 03 (GPU): feature extraction, HiFi-GAN copy-synthesis check (G1), training smoke test
and throughput measurement (G2), for both languages.

Inputs: outputs of fs2-01-ingest (raw_data_<lang>.tar, manifests/<lang>/) and fs2-02-align
(TextGrid_<lang>.tar). Outputs (/kaggle/working), a handful of files:
  preprocessed_<lang>.tar      mel/ pitch/ energy/ duration/ train.txt val.txt speakers.json stats.json
                               preprocess_summary.json (TextGrids are not repeated)
  copy_synthesis_<lang>.json   per-speaker vocoder distortion (lower is better)
  copy_synthesis_wavs_<lang>.tar
  smoke_<lang>/losses.csv, smoke_<lang>/run_*.json
  status.json, environment.txt, code_commit.txt, pytest.txt
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

OUT = "/kaggle/working"
TMP = "/kaggle/temp"
REPO = "https://github.com/ashdev-7/iiith-fastspeech2.git"
REPO_REF = os.environ.get("FS2_REF", "main")
CONFIG = {"te": "Telugu/config.yaml", "en": "English/config.yaml"}
SMOKE_STEPS = 400


def run(args, env=None, **kw):
    print("+", " ".join(args), flush=True)
    return subprocess.run(args, check=True, env=env, **kw)


def find_one(pattern, what):
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        listing = subprocess.run(["find", "/kaggle/input", "-maxdepth", "4"], capture_output=True, text=True).stdout
        raise FileNotFoundError(f"{what} not found ({pattern}). /kaggle/input:\n{listing[:3000]}")
    return hits[0]


def clone_code():
    dst = f"{TMP}/repo"
    run(["git", "clone", "-q", REPO, dst])
    run(["git", "-C", dst, "checkout", "-q", REPO_REF])
    commit = subprocess.run(["git", "-C", dst, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    with open(f"{OUT}/code_commit.txt", "w") as f:
        f.write(commit + "\n")
    return dst, commit


def stage_inputs(lang, root):
    """<root>/raw_data/<lang>, <root>/manifests/<lang>, <root>/preprocessed/<lang>/TextGrid."""
    os.makedirs(f"{root}/raw_data", exist_ok=True)
    run(["tar", "-xf", find_one(f"/kaggle/input/**/raw_data_{lang}.tar", f"raw_data_{lang}.tar"), "-C", f"{root}/raw_data"])
    man = os.path.dirname(find_one(f"/kaggle/input/**/manifests/{lang}/train.tsv", f"{lang} manifests"))
    shutil.copytree(man, f"{root}/manifests/{lang}", dirs_exist_ok=True)
    tg_parent = f"{root}/preprocessed/{lang}"
    os.makedirs(tg_parent, exist_ok=True)
    run(["tar", "-xf", find_one(f"/kaggle/input/**/TextGrid_{lang}.tar", f"TextGrid_{lang}.tar"), "-C", tg_parent])
    os.replace(f"{tg_parent}/{lang}", f"{tg_parent}/TextGrid")  # tar holds <lang>/<speaker>/<utt>.TextGrid


def smoke_config(repo, lang):
    import yaml
    with open(f"{repo}/{CONFIG[lang]}", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["train"]["step"].update({"total_step": SMOKE_STEPS, "log_step": 50, "val_step": 200, "save_step": SMOKE_STEPS})
    for k in cfg["train"]["path"]:
        cfg["train"]["path"][k] = cfg["train"]["path"][k].replace(f"output/{lang}", f"smoke/{lang}")
    path = f"{repo}/{os.path.dirname(CONFIG[lang])}/smoke_config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path


def main():
    run([sys.executable, "-m", "pip", "install", "-q", "tgt", "pyworld", "g2p_en", "unidecode", "inflect"])
    repo, commit = clone_code()
    src = f"{repo}/fastspeech2"
    env = dict(os.environ, PYTHONPATH=src, PYTHONIOENCODING="utf-8")
    with open(f"{OUT}/environment.txt", "w") as f:
        f.write(sys.version + "\n")
        f.write(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout)
        f.write(subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)
    status = {"code_commit": commit, "languages": {}}

    # Gate: full test suite, including the preprocessor test that is skipped on machines without torch.
    t = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "-m", "not observation", f"{repo}/tests"],
                       cwd=repo, env=env, capture_output=True, text=True)
    obs = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rA", "-m", "observation", f"{repo}/tests"],
                         cwd=repo, env=env, capture_output=True, text=True)
    with open(f"{OUT}/pytest.txt", "w") as f:
        f.write(t.stdout + t.stderr + "\n\n==== observations (not a gate) ====\n" + obs.stdout + obs.stderr)
    status["pytest"] = {"returncode": t.returncode, "tail": t.stdout.strip().splitlines()[-1:],
                        "observations_tail": obs.stdout.strip().splitlines()[-1:]}
    if t.returncode != 0:
        status["error"] = "tests failed; see pytest.txt"
        json.dump(status, open(f"{OUT}/status.json", "w"), indent=1)
        raise SystemExit(status["error"])

    for lang in ("te", "en"):
        root, st = f"{TMP}/data", {}
        status["languages"][lang] = st
        try:
            t0 = time.time()
            stage_inputs(lang, root)
            run([sys.executable, f"{src}/preprocess.py", "--config", f"{repo}/{CONFIG[lang]}", "--data_root", root], env)
            pre = f"{root}/preprocessed/{lang}"
            st["preprocess"] = json.load(open(f"{pre}/preprocess_summary.json"))
            st["preprocess"].pop("skipped", None)
            st["preprocess_minutes"] = round((time.time() - t0) / 60, 1)
            run(["tar", "-cf", f"{OUT}/preprocessed_{lang}.tar", "-C", pre, "--exclude=./TextGrid", "."])

            cs_dir = f"{TMP}/copy_synthesis/{lang}"
            run([sys.executable, f"{src}/copy_synthesis.py", "--config", f"{repo}/{CONFIG[lang]}", "--data_root", root,
                 "--out_dir", cs_dir, "--per_speaker", "20" if lang == "te" else "3", "--save_wavs"], env)
            shutil.copy(f"{cs_dir}/copy_synthesis.json", f"{OUT}/copy_synthesis_{lang}.json")
            run(["tar", "-cf", f"{OUT}/copy_synthesis_wavs_{lang}.tar", "-C", cs_dir, "."])
            st["copy_synthesis_ranked"] = json.load(open(f"{cs_dir}/copy_synthesis.json"))["ranked"][:5]

            t0 = time.time()
            run([sys.executable, f"{src}/train.py", "--config", smoke_config(repo, lang), "--data_root", root,
                 "--max_minutes", "20"], env)
            st["smoke_minutes"] = round((time.time() - t0) / 60, 1)
            smoke_log = f"{root}/smoke/{lang}/log"
            os.makedirs(f"{OUT}/smoke_{lang}", exist_ok=True)
            for p in glob.glob(f"{smoke_log}/*.csv") + glob.glob(f"{smoke_log}/run_*.json"):
                shutil.copy(p, f"{OUT}/smoke_{lang}/")
            rows = [l.split(",") for l in open(f"{smoke_log}/losses.csv").read().splitlines()[1:]]
            train_rows = [r for r in rows if r[0] == "train"]
            if len(train_rows) >= 2:  # steps/s between the first and last train log (excludes start-up)
                (s0, t_0), (s1, t_1) = (int(train_rows[0][1]), float(train_rows[0][-1])), (int(train_rows[-1][1]), float(train_rows[-1][-1]))
                st["train_steps_per_s_incl_val"] = round((s1 - s0) / (t_1 - t_0), 3)
            st["smoke_losses_first_last"] = [train_rows[0][3:9], train_rows[-1][3:9]] if train_rows else None
            st["ok"] = True
        except Exception as e:
            st["ok"] = False
            st["error"] = f"{type(e).__name__}: {e}"[:3000]
            traceback.print_exc()
        finally:
            shutil.rmtree(root, ignore_errors=True)
        json.dump(status, open(f"{OUT}/status.json", "w"), indent=1, ensure_ascii=False)

    print(json.dumps(status, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
