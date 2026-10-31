"""Load one per-language config.yaml into the (preprocess, model, train) dicts the reference code expects.

config.yaml has three top-level sections: preprocess, model, train. Paths in it are relative to a
data root (default: the directory holding config.yaml), so the same file works locally and on Kaggle:
    load_configs("English/config.yaml", data_root="/kaggle/temp/en")
The text language (preprocess.preprocessing.text.language) is copied into model["text_language"],
which transformer/Models.py uses to size the phone embedding table.
"""
import copy
import os

import yaml


def load_configs(config_path, data_root=None):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    missing = {"preprocess", "model", "train"} - set(cfg)
    if missing:
        raise KeyError(f"{config_path}: missing sections {sorted(missing)}")
    root = data_root or os.path.dirname(os.path.abspath(config_path))
    preprocess, model, train = (copy.deepcopy(cfg[k]) for k in ("preprocess", "model", "train"))
    for section in (preprocess, train):
        for key, value in section.get("path", {}).items():
            if isinstance(value, str) and not os.path.isabs(value):
                section["path"][key] = os.path.normpath(os.path.join(root, value))
    model["text_language"] = preprocess["preprocessing"]["text"]["language"]
    return preprocess, model, train
