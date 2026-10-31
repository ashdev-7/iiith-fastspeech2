"""Feature extraction (mel, phone-level pitch/energy, durations) for one language.

Expects <preprocessed_path>/TextGrid/<speaker>/<utt>.TextGrid from MFA, the wav/.lab files in
raw_path and train.tsv / val.tsv in split_path (see config.yaml).
  python preprocess.py --config ../Telugu/config.yaml --data_root /kaggle/temp/te_data
"""
import argparse

from preprocessor.preprocessor import Preprocessor
from utils.config import load_configs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    args = parser.parse_args()
    preprocess_config, _, _ = load_configs(args.config, args.data_root)
    Preprocessor(preprocess_config).build_from_path()
