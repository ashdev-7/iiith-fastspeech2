"""Generate the per-variant Kaggle job folders from their templates:
  kaggle/03_train_<lang>/  from kaggle/03_train/run_template.py   (@LANG@ = te | en)
  kaggle/04_eval_<mode>/   from kaggle/04_eval/run_template.py    (@MODE@ = smoke | full)
Push with:  kaggle kernels push -p kaggle/03_train_te --accelerator NvidiaTeslaT4
            kaggle kernels push -p kaggle/04_eval_smoke             (CPU)
            kaggle kernels push -p kaggle/04_eval_full --accelerator NvidiaTeslaT4
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "kaggle", "03_train", "run_template.py")
EVAL_TEMPLATE = os.path.join(ROOT, "kaggle", "04_eval", "run_template.py")


def _kernel(slug, gpu, sources):
    return {"id": f"ashdev07/{slug}", "title": slug, "code_file": "run.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": gpu,
            "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": [],
            "kernel_sources": sources}


def render(lang):
    with open(TEMPLATE, encoding="utf-8") as f:
        return f.read().replace("@LANG@", lang)


def metadata(lang):
    return _kernel(f"fs2-03-train-{lang}", True, ["ashdev07/fs2-02-data"])


def render_eval(mode):
    with open(EVAL_TEMPLATE, encoding="utf-8") as f:
        return f.read().replace("@MODE@", mode)


def metadata_eval(mode):
    if mode == "smoke":
        return _kernel("fs2-04-eval-smoke", False, ["ashdev07/fs2-02-data"])
    return _kernel("fs2-04-eval", True, ["ashdev07/fs2-02-data", "ashdev07/fs2-03-train-te",
                                         "ashdev07/fs2-03-train-en"])


def _write(d, code, meta):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "run.py"), "w", encoding="utf-8", newline="\n") as f:
        f.write(code)
    with open(os.path.join(d, "kernel-metadata.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")


def main():
    for lang in ("te", "en"):
        _write(os.path.join(ROOT, "kaggle", f"03_train_{lang}"), render(lang), metadata(lang))
    for mode in ("smoke", "full"):
        _write(os.path.join(ROOT, "kaggle", f"04_eval_{mode}"), render_eval(mode), metadata_eval(mode))


if __name__ == "__main__":
    main()
