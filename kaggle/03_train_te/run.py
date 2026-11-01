"""Kaggle job 03 (GPU T4): train FastSpeech 2 for ONE language, then synthesise its test sentences.
Generated from kaggle/03_train/run_template.py by tools/make_train_jobs.py (only LANG differs).

Stages (kaggle/pipeline.py): tests -> unpack features (fs2-02-data) -> throughput probe ->
dry run of loss plots + inference on the probe checkpoint (fail fast) -> training sized from the
probe (total_step, LR decay) with a time limit -> checkpoints saved -> training_loss.png,
validation_loss.png -> inference (5 wavs, 15 variance plots, variance JSON, RTF timing) with the
pre-registered test voice.
Outputs: status.json, pytest.txt, environment.txt, code_commit.txt,
  results_<lang>.tar (Results/<Lang>: audio, plots, variance_raw, loss plot, losses.csv, configs, stats),
  <lang>_model.pth.tar (weights only, for release), <lang>_full.pth.tar (with optimizer, for resuming),
  details.tar.
"""
import os
import shutil
import subprocess
import sys

LANG = "te"
TRAIN_BUDGET_MIN = 480  # training wall-clock limit (8 h of the 12 h session; 2 languages ~17 of 30 weekly GPU h)
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

LANG_DIR = {"te": "Telugu", "en": "English"}[LANG]
root = f"{TMP}/data"
status = P.Status(job=f"train_{LANG}", code_commit=commit, lang=LANG)
try:
    P.setup_python(status)
    if not status.data["stages"]["environment"].get("cuda_matmul_ok"):
        raise SystemExit("no working GPU in this session")
    P.run_tests(status, repo)
    with status.stage("telugu_font", fatal=False) as rec:
        font_dir = rec["font_dir"] = P.fetch_telugu_font()
    with status.stage("stage_inputs") as rec:
        P.unpack_preprocessed(LANG, root)
        speaker = rec["test_voice"] = P.data_job_test_voice(LANG)
    with status.stage("throughput_probe") as rec:
        cfg = P.derived_config(repo, LANG, "probe", {"train": {
            "step": {"total_step": 300, "log_step": 50, "val_step": 100000, "save_step": 100000},
            "path": {"ckpt_path": f"probe/{LANG}/ckpt", "log_path": f"probe/{LANG}/log",
                     "result_path": f"probe/{LANG}/result"}}})
        probe_ckpt, probe_log = P.train(LANG, root, repo, rec, cfg, "15", restore="0")
        plan = rec["plan"] = P.size_training(rec["steps_per_s"], TRAIN_BUDGET_MIN)
    # Dry run of everything that happens AFTER the long training (loss plots, inference on the GPU,
    # vocoder, fonts, plot layout) on the 300-step probe checkpoint, so a bug there fails the job in
    # minutes instead of after 8 hours. Outputs are discarded (untrained model).
    with status.stage("post_training_dry_run") as rec:
        dry = f"{TMP}/dry_run"
        P.loss_plot(repo, [probe_log], dry, "dry run")
        P.inference(LANG, root, repo, rec, probe_ckpt, speaker, dry, font_dir)
        shutil.rmtree(dry, ignore_errors=True)
        shutil.rmtree(f"{root}/probe", ignore_errors=True)
    with status.stage("train") as rec:
        cfg = P.derived_config(repo, LANG, "final", {"train": {
            "step": {k: plan[k] for k in ("total_step", "val_step", "save_step", "log_step")},
            "optimizer": {"anneal_steps": plan["anneal_steps"]}}})
        ckpt, log_dir = P.train(LANG, root, repo, rec, cfg, str(TRAIN_BUDGET_MIN), restore="0")
        rec["checkpoint"] = os.path.basename(ckpt)
    with status.stage("checkpoints") as rec:  # saved before anything else can fail
        P.model_only_checkpoint(ckpt, f"{OUT}/{LANG}_model.pth.tar")
        shutil.copy(ckpt, f"{OUT}/{LANG}_full.pth.tar")
    results = f"{OUT}/Results/{LANG_DIR}"
    os.makedirs(results, exist_ok=True)
    with status.stage("loss_plot", fatal=False) as rec:
        for f in ("losses.csv", "val_diagnostics.csv"):
            if os.path.exists(f"{log_dir}/{f}"):
                shutil.copy(f"{log_dir}/{f}", results)
        rec["png"] = P.loss_plot(repo, [log_dir], results, f"{LANG_DIR} FastSpeech 2")
    with status.stage("inference", fatal=False) as rec:
        rec["speaker"] = speaker
        P.inference(LANG, root, repo, rec, ckpt, speaker, results, font_dir)
    with status.stage("package") as rec:
        for f in ("stats.json", "speakers.json"):
            shutil.copy(f"{root}/preprocessed/{LANG}/{f}", results)
        shutil.copy(cfg, f"{results}/train_config_used.yaml")
        P.tar(f"{OUT}/results_{LANG}.tar", OUT, "Results")
        shutil.rmtree(f"{OUT}/Results")
        rec["files"] = sorted(os.listdir(OUT))
finally:
    status.data["outputs"] = P.finalize()
    status.data["failed_stages"] = [k for k, v in status.data["stages"].items() if v.get("ok") is False]
    status.data["all_ok"] = not status.data["failed_stages"]
    status.write()
