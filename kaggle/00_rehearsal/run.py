"""Kaggle job 00 (CPU): end-to-end REHEARSAL on a small real subset, through the same pipeline
functions as the full runs (kaggle/pipeline.py): tests -> MFA -> features -> copy-synthesis ->
training (+ resume) -> inference (10 wavs, 30 plots) -> loss plot. Stage timings are used to size the
full runs. Input: fs2-01-ingest output. Outputs: status.json, pytest.txt, environment.txt,
code_commit.txt, results.tar (wavs, plots, JSON), details.tar (MFA logs/help, configs, everything else).
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

SUBSET = {"te": dict(n_train=40, n_val=8, max_speakers=2), "en": dict(n_train=25, n_val=5, max_speakers=4)}
TRAIN = {"optimizer": {"batch_size": 4, "warm_up_step": 10, "anneal_steps": [1000, 2000, 3000]},
         "step": {"log_step": 5, "val_step": 10, "save_step": 15}}

status = P.Status(job="rehearsal", code_commit=commit)
try:
    P.setup_python(status)
    P.run_tests(status, repo)
    P.install_mfa(status)
    with status.stage("telugu_font", fatal=False) as rec:
        font_dir = P.fetch_telugu_font()
        rec["font_dir"] = font_dir
    for lang in ("te", "en"):
        root = f"{TMP}/data"
        with status.stage(f"{lang}_stage_inputs", fatal=False) as rec:  # e.g. ingest output missing a language
            P.unpack_ingest(lang, root)
            rec.update(P.subset_manifests(root, lang, **SUBSET[lang]))
        if not status.ok(f"{lang}_stage_inputs"):
            continue
        with status.stage(f"{lang}_align", fatal=False) as rec:
            P.align(lang, root, repo, rec)
        P.save_mfa_logs(lang)
        if not status.ok(f"{lang}_align"):
            continue
        with status.stage(f"{lang}_preprocess", fatal=False) as rec:
            P.preprocess(lang, root, repo, rec)
        if not status.ok(f"{lang}_preprocess"):
            continue
        cs_json = None
        with status.stage(f"{lang}_copy_synthesis", fatal=False) as rec:
            cs_json = P.copy_synthesis(lang, root, repo, f"{OUT}/results/{lang}/copy_synthesis", "3", rec)
        rehearsal_paths = {"path": {k: f"rehearsal/{lang}/{k.split('_')[0]}" for k in ("ckpt_path", "log_path", "result_path")}}
        ckpt = log_dir = None
        with status.stage(f"{lang}_train", fatal=False) as rec:
            cfg = P.derived_config(repo, lang, "rehearsal_a", {"train": dict(TRAIN, step=dict(TRAIN["step"], total_step=30), **rehearsal_paths)})
            ckpt, log_dir = P.train(lang, root, repo, rec, cfg, "20", restore="0")
        with status.stage(f"{lang}_train_resume", fatal=False) as rec:
            cfg = P.derived_config(repo, lang, "rehearsal_b", {"train": dict(TRAIN, step=dict(TRAIN["step"], total_step=40), **rehearsal_paths)})
            ckpt, log_dir = P.train(lang, root, repo, rec, cfg, "20", restore="latest")
            if rec["start_step"] != 30 or rec["end_step"] != 40:
                raise RuntimeError(f"resume went {rec['start_step']} -> {rec['end_step']}, expected 30 -> 40")
        if ckpt:
            with status.stage(f"{lang}_loss_plot", fatal=False) as rec:
                rec["png"] = P.loss_plot(repo, [log_dir], f"{OUT}/results/{lang}/loss_curves.png", f"{lang} rehearsal")
            with status.stage(f"{lang}_inference", fatal=False) as rec:
                speaker = P.test_voice(root, lang, cs_json)
                rec["speaker"] = speaker
                P.inference(lang, root, repo, rec, ckpt, speaker, f"{OUT}/results/{lang}", font_dir)
        subprocess.run(["rm", "-rf", root])
finally:
    try:
        P.tar(f"{OUT}/results.tar", OUT, "results")
        subprocess.run(["rm", "-rf", f"{OUT}/results"])
    except Exception as e:  # noqa: BLE001
        print("results tar failed:", e)
    status.data["outputs"] = P.finalize()
    failed = [k for k, v in status.data["stages"].items() if v.get("ok") is False]
    status.data["failed_stages"] = failed
    status.data["all_ok"] = not failed
    status.write()
    print(status.data)
