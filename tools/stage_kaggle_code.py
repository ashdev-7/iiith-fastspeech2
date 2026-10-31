"""Copy the project code into a staging folder for the private Kaggle dataset "<user>/fs2-code".

Usage:
  python tools/stage_kaggle_code.py --user ashdev07 --out <staging dir> [--with_vocoder]
  kaggle datasets create -p <staging dir> --dir-mode zip      # first time
  kaggle datasets version -p <staging dir> -m "msg" --dir-mode zip
"""
import argparse
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDE = ["fastspeech2", "tests/test_sentences.json"]
SKIP_DIRS = {"__pycache__", ".pytest_cache"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--with_vocoder", action="store_true")
    args = ap.parse_args()

    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out)
    for rel in INCLUDE:
        src = os.path.join(ROOT, rel)
        dst = os.path.join(args.out, "project", rel)
        if os.path.isdir(src):
            ignore = shutil.ignore_patterns(*SKIP_DIRS, *([] if args.with_vocoder else ["*.pth.tar.zip"]))
            shutil.copytree(src, dst, ignore=ignore)
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
    with open(os.path.join(args.out, "dataset-metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"title": "fs2-code", "id": f"{args.user}/fs2-code",
                   "licenses": [{"name": "other"}]}, f, indent=1)
    print("staged", args.out)


if __name__ == "__main__":
    main()
