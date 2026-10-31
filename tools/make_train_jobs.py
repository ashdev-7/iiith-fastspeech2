"""Generate kaggle/03_train_<lang>/{run.py,kernel-metadata.json} from kaggle/03_train/run_template.py.
Push with:  kaggle kernels push -p kaggle/03_train_te --accelerator NvidiaTeslaT4
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(ROOT, "kaggle", "03_train", "run_template.py")


def render(lang):
    with open(TEMPLATE, encoding="utf-8") as f:
        return f.read().replace("@LANG@", lang)


def metadata(lang):
    return {"id": f"ashdev07/fs2-03-train-{lang}", "title": f"fs2-03-train-{lang}", "code_file": "run.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
            "enable_tpu": False, "enable_internet": True, "dataset_sources": [], "competition_sources": [],
            "kernel_sources": ["ashdev07/fs2-02-data"]}


def main():
    for lang in ("te", "en"):
        d = os.path.join(ROOT, "kaggle", f"03_train_{lang}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "run.py"), "w", encoding="utf-8", newline="\n") as f:
            f.write(render(lang))
        with open(os.path.join(d, "kernel-metadata.json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump(metadata(lang), f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
