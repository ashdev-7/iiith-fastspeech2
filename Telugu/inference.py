"""Telugu model: runs fastspeech2/inference.py with this folder's config.yaml (other arguments are passed through).
  python Telugu/inference.py --data_root <data root>
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(os.path.dirname(HERE), "fastspeech2")
sys.path.insert(0, CODE)
if "--config" not in sys.argv:
    sys.argv[1:1] = ["--config", os.path.join(HERE, "config.yaml")]
sys.argv[0] = os.path.join(CODE, "inference.py")
runpy.run_path(sys.argv[0], run_name="__main__")
