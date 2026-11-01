"""Shared stages for the Kaggle jobs. Every job (rehearsal and full runs) calls these same functions,
so a passing rehearsal exercises exactly the code paths of the full runs.

Each job's run.py clones the repository, puts <repo>/kaggle on sys.path and imports this module.
API-budget rule: every job leaves only a handful of files in /kaggle/working (tars + small JSON),
and status.json is rewritten after every stage, so one small fetch tells what happened.
"""
import contextlib
import csv
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
MAMBA = f"{TMP}/bin/micromamba"
MFA_ENV = f"{TMP}/mfa_env"
CONFIG = {"te": "Telugu/config.yaml", "en": "English/config.yaml"}
PIP = ["tgt", "pyworld", "g2p_en", "unidecode", "inflect", "pytest"]
FONT_URLS = [  # Noto Sans Telugu (SIL OFL 1.1)
    "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSansTelugu/hinted/ttf/NotoSansTelugu-Regular.ttf",
    "https://github.com/google/fonts/raw/main/ofl/notosanstelugu/NotoSansTelugu%5Bwdth%2Cwght%5D.ttf",
]


# ----------------------------------------------------------------------------- bookkeeping
class Status:
    def __init__(self, path=f"{OUT}/status.json", **info):
        self.path = path
        self.data = dict(info, stages={}, started=time.strftime("%Y-%m-%d %H:%M:%S"))
        self.write()

    def write(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=1, ensure_ascii=False)
        os.replace(tmp, self.path)

    @contextlib.contextmanager
    def stage(self, name, fatal=True):
        """Times a stage and records ok/error. fatal=False: the error is recorded and the job goes on."""
        rec = {"ok": None}
        self.data["stages"][name] = rec
        self.write()
        t0 = time.time()
        print(f"\n===== {name} =====", flush=True)
        try:
            yield rec
            rec["ok"] = True
        except BaseException as e:  # noqa: B036  (SystemExit from failed checks must be recorded too)
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {e}"[-3000:]
            rec["traceback_tail"] = traceback.format_exc()[-3000:]
            traceback.print_exc()
            if fatal:
                rec["minutes"] = round((time.time() - t0) / 60, 2)
                self.data["fatal_stage"] = name
                self.write()
                raise
        finally:
            rec["minutes"] = round((time.time() - t0) / 60, 2)
            self.write()

    def ok(self, name):
        return self.data["stages"].get(name, {}).get("ok") is True


def run(args, env=None, cwd=None, log=None):
    """Run a command; on failure keep the last lines of its output in the exception message."""
    print("+", " ".join(map(str, args)), flush=True)
    p = subprocess.run(list(map(str, args)), env=env, cwd=cwd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    print(out[-20000:], flush=True)
    if log:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write("+ " + " ".join(map(str, args)) + "\n" + out + "\n")
    if p.returncode != 0:
        raise RuntimeError(f"exit {p.returncode}: {' '.join(map(str, args))[:300]}\n...{out[-2500:]}")
    return out


def optional_output(args):
    """Output of a diagnostic/optional command, or a note if the tool is missing or fails.
    Never raises (e.g. nvidia-smi does not exist on CPU machines)."""
    try:
        p = subprocess.run(list(map(str, args)), capture_output=True, text=True, timeout=600)
        return p.stdout if p.returncode == 0 else f"[{args[0]} exited {p.returncode}] {p.stderr[-500:]}"
    except (OSError, subprocess.SubprocessError) as e:
        return f"[{args[0]} unavailable: {type(e).__name__}: {e}]"


def find_one(pattern, what):
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        listing = optional_output(["find", "/kaggle/input", "-maxdepth", "4"])
        raise FileNotFoundError(f"{what} not found ({pattern}). /kaggle/input:\n{listing[:2500]}")
    return hits[0]


def code_env(repo):
    return dict(os.environ, PYTHONPATH=f"{repo}/fastspeech2", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")


# ----------------------------------------------------------------------------- environment
def setup_python(status):
    with status.stage("pip_install") as rec:
        failed = []
        for pkg in PIP:  # one by one, so a failure names the package
            try:
                run([sys.executable, "-m", "pip", "install", "-q", pkg])
            except RuntimeError:
                try:  # e.g. pyworld: build against the installed numpy/Cython instead of an isolated env
                    run([sys.executable, "-m", "pip", "install", "-q", "cython", "setuptools", "wheel"])
                    run([sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", pkg])
                    rec.setdefault("fallback_used", []).append(pkg)
                except RuntimeError:
                    failed.append(pkg)
        if failed:
            raise RuntimeError(f"could not install {failed}")
        # Import check now, not hours later.
        run([sys.executable, "-c", "import tgt, pyworld, g2p_en, librosa, sklearn, soundfile, scipy, matplotlib, yaml"])
    with status.stage("environment") as rec:
        import torch
        rec["python"] = sys.version.split()[0]
        rec["torch"] = torch.__version__
        rec["cuda_available"] = torch.cuda.is_available()
        rec["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available():  # fails here (not hours later) if the GPU arch is unsupported by this torch
            x = torch.randn(256, 256, device="cuda")
            rec["cuda_matmul_ok"] = bool(torch.isfinite(x @ x).all().item())
            rec["cuda_arch_list"] = torch.cuda.get_arch_list()
        with open(f"{OUT}/environment.txt", "w") as f:
            f.write(sys.version + "\n" + json.dumps(rec) + "\n")
            f.write(optional_output(["nvidia-smi"]))
            f.write(subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)


def run_tests(status, repo):
    env = code_env(repo)
    with status.stage("pytest_gate") as rec:
        p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "-m", "not observation", "tests"],
                           cwd=repo, env=env, capture_output=True, text=True)
        o = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rA", "-m", "observation", "tests"],
                           cwd=repo, env=env, capture_output=True, text=True)
        with open(f"{OUT}/pytest.txt", "w", encoding="utf-8") as f:
            f.write(p.stdout + p.stderr + "\n\n==== observations (not a gate) ====\n" + o.stdout + o.stderr)
        rec["summary"] = p.stdout.strip().splitlines()[-1:]
        rec["observations"] = o.stdout.strip().splitlines()[-1:]
        if p.returncode != 0:
            raise RuntimeError("tests failed; see pytest.txt")


def fetch_telugu_font(dst=f"{TMP}/fonts"):
    """Returns the directory holding a Telugu font, or None. Tries the system, URLs, then apt."""
    os.makedirs(dst, exist_ok=True)
    for url in FONT_URLS:
        path = os.path.join(dst, "NotoSansTelugu.ttf")
        optional_output(["curl", "-fsSL", "-o", path, url])
        if os.path.exists(path) and os.path.getsize(path) > 50000:
            return dst
    if not optional_output(["bash", "-c", "apt-get install -y -q fonts-noto-core >/dev/null 2>&1"]).startswith("["):
        hits = glob.glob("/usr/share/fonts/**/NotoSansTelugu*", recursive=True)
        if hits:
            return os.path.dirname(hits[0])
    return None


# ----------------------------------------------------------------------------- data staging
def unpack_ingest(lang, root):
    """<root>/raw_data/<lang>/<speaker>/<utt>.wav|.lab and <root>/manifests/<lang>/*.tsv"""
    os.makedirs(f"{root}/raw_data", exist_ok=True)
    if not os.path.isdir(f"{root}/raw_data/{lang}"):
        run(["tar", "-xf", find_one(f"/kaggle/input/**/raw_data_{lang}.tar", f"raw_data_{lang}.tar"),
             "-C", f"{root}/raw_data"])
    man = os.path.dirname(find_one(f"/kaggle/input/**/manifests/{lang}/train.tsv", f"{lang} manifests"))
    shutil.copytree(man, f"{root}/manifests/{lang}", dirs_exist_ok=True)


def read_tsv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def subset_manifests(root, lang, n_train, n_val, max_speakers):
    """Rehearsal only: keep n_train/n_val utterances for each of the max_speakers speakers with the
    most training utterances among those that also have validation utterances."""
    man = f"{root}/manifests/{lang}"
    train, val = read_tsv(f"{man}/train.tsv"), read_tsv(f"{man}/val.tsv")
    val_spk = {r["speaker"] for r in val}
    counts = {}
    for r in train:
        if r["speaker"] in val_spk:
            counts[r["speaker"]] = counts.get(r["speaker"], 0) + 1
    speakers = sorted(counts, key=lambda s: (-counts[s], s))[:max_speakers]
    sub = {}
    for name, rows, n in (("train", train, n_train), ("val", val, n_val)):
        keep = []
        for s in speakers:
            keep += sorted((r for r in rows if r["speaker"] == s), key=lambda r: float(r["duration"]))[:n]
        write_tsv(f"{man}/{name}.tsv", keep)
        sub[name] = len(keep)
    return {"speakers": speakers, **sub}


def test_voice(root, lang, copy_synthesis_json=None):
    """Pre-registered rule (plan.txt REVISION 1): en = speaker with the most training audio;
    te = speaker with the lower HiFi-GAN copy-synthesis distortion."""
    if lang == "te" and copy_synthesis_json and os.path.exists(copy_synthesis_json):
        return json.load(open(copy_synthesis_json, encoding="utf-8"))["ranked"][0]
    hours = {}
    for r in read_tsv(f"{root}/manifests/{lang}/train.tsv"):
        hours[r["speaker"]] = hours.get(r["speaker"], 0.0) + float(r["duration"])
    return sorted(hours, key=lambda s: (-hours[s], s))[0]


# ----------------------------------------------------------------------------- alignment
def install_mfa(status):
    with status.stage("install_mfa") as rec:
        os.makedirs(TMP, exist_ok=True)
        if not os.path.exists(MAMBA):
            run(["bash", "-c", f"cd {TMP} && curl -fLs https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba"])
        if not os.path.isdir(MFA_ENV):
            run([MAMBA, "create", "-y", "-q", "-p", MFA_ENV, "-c", "conda-forge", "montreal-forced-aligner"])
        rec["version"] = mfa("version").strip()[-200:]
        rec["runs_as"] = MFA_USER if _as_root() else "current user"
        # Keep the CLI help so flag errors can be diagnosed from the status alone.
        for cmd in ("align", "train"):
            h = mfa(cmd, "--help")
            with open(f"{OUT}/mfa_{cmd}_help.txt", "w") as f:
                f.write(h)


MFA_USER = "mfa"
MFA_HOME = f"/home/{MFA_USER}"


def _as_root():
    return hasattr(os, "geteuid") and os.geteuid() == 0


def mfa_writable(*dirs):
    """MFA runs as a non-root user (PostgreSQL, which MFA 3 uses, refuses to run as root;
    the official MFA Docker image also runs as a non-root user). Make its output dirs writable."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        if _as_root():
            run(["chmod", "-R", "a+rwX", d])


def mfa(*args, log=None):
    cmd = [MAMBA, "run", "-p", MFA_ENV, "mfa", *args]
    if _as_root():
        if subprocess.run(["id", MFA_USER], capture_output=True).returncode != 0:
            run(["useradd", "-m", "-d", MFA_HOME, MFA_USER])
        cmd = ["runuser", "-u", MFA_USER, "--", "env", f"HOME={MFA_HOME}", f"MFA_ROOT_DIR={MFA_HOME}/MFA",
               f"MAMBA_ROOT_PREFIX={MFA_HOME}/.mamba"] + cmd
    return run(cmd, log=log)


def align(lang, root, repo, rec, jobs=None):
    """MFA on train+val of <root>/manifests/<lang>; TextGrids -> <root>/preprocessed/<lang>/TextGrid."""
    jobs = str(jobs or os.cpu_count() or 4)
    corpus, tg = f"{TMP}/corpus_{lang}", f"{root}/preprocessed/{lang}/TextGrid"
    shutil.rmtree(corpus, ignore_errors=True)
    shutil.rmtree(tg, ignore_errors=True)
    os.makedirs(f"{OUT}/mfa_{lang}", exist_ok=True)
    dict_path = f"{OUT}/mfa_{lang}/dictionary.txt"
    log = f"{OUT}/mfa_{lang}/commands.log"
    run([sys.executable, f"{repo}/fastspeech2/data/mfa_prepare.py", "--lang", lang, "--ingest_dir", root,
         "--corpus", corpus, "--dict_out", dict_path], env=code_env(repo), log=log)
    info = json.load(open(f"{OUT}/mfa_{lang}/dictionary_info.json", encoding="utf-8"))
    rec["dictionary_words"] = info["dictionary_words"]
    rec["words_without_pronunciation"] = len(info["words_without_pronunciation"])
    mfa_writable(os.path.dirname(tg), f"{TMP}/mfa_tmp_{lang}", f"{OUT}/mfa_{lang}")
    common = ["--clean", "-j", jobs, "--temporary_directory", f"{TMP}/mfa_tmp_{lang}"]
    if lang == "en":
        mfa("model", "download", "acoustic", "english_us_arpa", log=log)
        mfa("align", corpus, dict_path, "english_us_arpa", tg, "--beam", "100", "--retry_beam", "400", *common, log=log)
    else:
        mfa("train", corpus, dict_path, f"{OUT}/mfa_{lang}/telugu_acoustic_model.zip",
            "--output_directory", tg, *common, log=log)
    expected = {os.path.relpath(p, corpus)[:-4] for p in glob.glob(f"{corpus}/*/*.lab")}
    got = {os.path.relpath(p, tg)[:-9] for p in glob.glob(f"{tg}/*/*.TextGrid")}
    rec.update({"expected": len(expected), "aligned": len(expected & got), "missing": len(expected - got),
                "missing_examples": sorted(expected - got)[:20]})
    if not expected & got:
        raise RuntimeError("MFA produced no TextGrids")
    return tg


def save_mfa_logs(lang):
    for p in glob.glob(f"{TMP}/mfa_tmp_{lang}/**/*.log", recursive=True)[:80]:
        dst = f"{OUT}/mfa_{lang}/logs/{os.path.relpath(p, f'{TMP}/mfa_tmp_{lang}')}"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(p, dst)


# ----------------------------------------------------------------------------- model stages
def derived_config(repo, lang, name, overrides):
    """Write <repo>/<Lang>/<name>.yaml = config.yaml deep-merged with overrides; return its path."""
    import yaml

    def merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                merge(a[k], v)
            else:
                a[k] = v
        return a
    with open(f"{repo}/{CONFIG[lang]}", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    merge(cfg, overrides)
    path = f"{repo}/{os.path.dirname(CONFIG[lang])}/{name}.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path


def preprocess(lang, root, repo, rec, config=None):
    config = config or f"{repo}/{CONFIG[lang]}"
    run([sys.executable, f"{repo}/fastspeech2/preprocess.py", "--config", config, "--data_root", root], env=code_env(repo))
    s = json.load(open(f"{root}/preprocessed/{lang}/preprocess_summary.json", encoding="utf-8"))
    rec.update({k: v for k, v in s.items() if k != "skipped"})
    if not s["utterances"]["train"] or not s["utterances"]["val"]:
        raise RuntimeError("preprocessing produced an empty split")


def copy_synthesis(lang, root, repo, out_dir, per_speaker, rec):
    run([sys.executable, f"{repo}/fastspeech2/copy_synthesis.py", "--config", f"{repo}/{CONFIG[lang]}",
         "--data_root", root, "--out_dir", out_dir, "--per_speaker", per_speaker, "--save_wavs"], env=code_env(repo))
    cs = json.load(open(f"{out_dir}/copy_synthesis.json", encoding="utf-8"))
    rec["ranked"] = cs["ranked"][:5]
    rec["per_speaker"] = {s: cs["per_speaker"][s] for s in cs["ranked"][:5]}
    return f"{out_dir}/copy_synthesis.json"


def train(lang, root, repo, rec, config, max_minutes, restore="latest", extra=()):
    run([sys.executable, f"{repo}/fastspeech2/train.py", "--config", config, "--data_root", root,
         "--restore_step", restore, "--max_minutes", max_minutes, *extra], env=code_env(repo))
    import yaml
    paths = yaml.safe_load(open(config, encoding="utf-8"))["train"]["path"]
    log_dir, ckpt_dir = os.path.join(root, paths["log_path"]), os.path.join(root, paths["ckpt_path"])
    runs = sorted(glob.glob(f"{log_dir}/run_*.json"), key=os.path.getmtime)
    info = json.load(open(runs[-1]))
    rec.update({k: info[k] for k in ("start_step", "end_step", "stop_reason", "minutes", "gpus", "num_params")})
    rows = [r for r in read_csv_rows(f"{log_dir}/losses.csv") if int(r["step"]) > rec["start_step"]]  # this run only
    tr = [r for r in rows if r["split"] == "train"]
    if len(tr) >= 2:
        a, b = tr[0], tr[-1]
        rec["steps_per_s"] = round((int(b["step"]) - int(a["step"])) / max(1e-6, float(b["time_s"]) - float(a["time_s"])), 3)
        rec["train_total_loss_first_last"] = [float(a["total"]), float(b["total"])]
    va = [r for r in rows if r["split"] == "val"]
    if va:
        rec["val_total_loss_last"] = float(va[-1]["total"])
    ckpts = sorted(glob.glob(f"{ckpt_dir}/*.pth.tar"), key=lambda p: int(os.path.basename(p).split(".")[0]))
    if not ckpts:
        raise RuntimeError("no checkpoint written")
    return ckpts[-1], log_dir


def read_csv_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def inference(lang, root, repo, rec, checkpoint, speaker, out_dir, font_dir):
    args = [sys.executable, f"{repo}/fastspeech2/inference.py", "--config", f"{repo}/{CONFIG[lang]}",
            "--data_root", root, "--checkpoint", checkpoint, "--speaker", speaker, "--out_dir", out_dir]
    if font_dir:
        args += ["--font_dir", font_dir]
    run(args, env=code_env(repo))
    rep = json.load(open(f"{out_dir}/inference_report.json", encoding="utf-8"))
    wavs = glob.glob(f"{out_dir}/test_generated_audios/*.wav")
    pngs = glob.glob(f"{out_dir}/variance_adaptor_plots/*/*.png")
    rec.update({"wavs": len(wavs), "plots": len(pngs), "telugu_font": rep["telugu_font"],
                "title_rendering": rep.get("title_rendering"), "device": rep.get("device"),
                "timing_summary": rep.get("timing_summary"),
                "seconds": {k: v["seconds"] for k, v in rep["items"].items()},
                "zero_duration_phones": {k: v["checks"]["zero_duration_phones"] for k, v in rep["items"].items()}})
    if len(wavs) != 5 or len(pngs) != 15:
        raise RuntimeError(f"expected 5 wavs and 15 plots, got {len(wavs)} and {len(pngs)}")


def loss_plot(repo, log_dirs, out_dir, name):
    """Writes the assignment's training_loss.png and validation_loss.png, plus an overlay
    (loss_curves.png) for judging over/under-fitting. Returns the three paths."""
    sys.path.insert(0, f"{repo}/fastspeech2")
    from plotting import plot_losses
    csvs = [f"{d}/losses.csv" for d in log_dirs if os.path.exists(f"{d}/losses.csv")]
    return [plot_losses(csvs, f"{out_dir}/training_loss.png", f"{name}: training losses", ("train",)),
            plot_losses(csvs, f"{out_dir}/validation_loss.png", f"{name}: validation losses", ("val",)),
            plot_losses(csvs, f"{out_dir}/loss_curves.png", f"{name}: training vs validation losses")]


def tar(dst, src_dir, *members, exclude=()):
    run(["tar", "-cf", dst, "-C", src_dir, *[f"--exclude={e}" for e in exclude], *(members or ["."])])


def finalize(keep=("status.json", "pytest.txt", "environment.txt", "code_commit.txt"), bundle="details.tar"):
    """Pack everything in /kaggle/working except `keep` and existing tars into one tar, so the job
    output stays a handful of files (one cheap listing, no paging)."""
    rest = [n for n in os.listdir(OUT) if n not in keep and not n.endswith(".tar") and n != bundle]
    if rest:
        run(["tar", "-cf", f"{OUT}/{bundle}", "-C", OUT, *sorted(rest)])
        for n in rest:
            p = f"{OUT}/{n}"
            shutil.rmtree(p) if os.path.isdir(p) and not os.path.islink(p) else os.remove(p)
    return sorted(os.listdir(OUT))


# ----------------------------------------------------------------------------- GPU training
def unpack_preprocessed(lang, root):
    """<root>/preprocessed/<lang>/{mel,pitch,energy,duration,train.txt,val.txt,stats.json,speakers.json}"""
    dst = f"{root}/preprocessed/{lang}"
    os.makedirs(dst, exist_ok=True)
    run(["tar", "-xf", find_one(f"/kaggle/input/**/preprocessed_{lang}.tar", f"preprocessed_{lang}.tar"), "-C", dst])
    for f in ("train.txt", "val.txt", "stats.json", "speakers.json"):
        if not os.path.exists(f"{dst}/{f}"):
            raise FileNotFoundError(f"{dst}/{f} missing after unpacking")
    return dst


def data_job_test_voice(lang, status_glob="/kaggle/input/**/fs2-02-data/**/status.json"):
    """Test voice chosen by fs2-02-data with the pre-registered rule, read from its status.json."""
    st = json.load(open(find_one(status_glob, "fs2-02-data status.json"), encoding="utf-8"))
    return st["stages"][f"{lang}_test_voice"]["speaker"]


def size_training(steps_per_s, budget_minutes, cap=160000, floor=20000):
    """total_step from the measured throughput (5 % margin); learning-rate decay at the reference's
    proportions (300k/400k/500k of 900k)."""
    total = int(steps_per_s * budget_minutes * 60 * 0.95) // 1000 * 1000
    total = max(floor, min(cap, total))
    anneal = [int(total * f) // 1000 * 1000 for f in (300 / 900, 400 / 900, 500 / 900)]
    return {"total_step": total, "anneal_steps": anneal,
            "val_step": max(1000, total // 50 // 1000 * 1000), "save_step": 10000, "log_step": 100}


def model_only_checkpoint(full_ckpt, dst):
    """Release checkpoint: model weights + step, without the optimizer state."""
    import torch
    ck = torch.load(full_ckpt, map_location="cpu", weights_only=True)
    torch.save({"model": ck["model"], "step": ck.get("step")}, dst)
    return dst
