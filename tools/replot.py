"""Redraw the variance-adaptor and loss plots from the saved numbers, without the model.

Inputs: Results/<Lang>/variance_raw/<ID>.json (the plotted values) and Results/<Lang>/losses.csv.
Used to restyle the figures (greyscale) after the Kaggle runs; the numbers are unchanged.

Telugu titles need text shaping (conjuncts, vowel signs). On Kaggle this is done by Pillow with libraqm.
Where Pillow has no raqm (e.g. Windows), --browser renders the title with a headless Chromium/Edge,
which shapes Telugu with HarfBuzz, and the cropped image is used instead.

  python tools/replot.py --lang Telugu --browser "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
"""
import argparse
import glob
import html
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fastspeech2"))
import plotting  # noqa: E402


def browser_shaped_text_image(browser):
    def render(text, font_name, size_px=30):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            page = os.path.join(d, "t.html")
            png = os.path.join(d, "t.png")
            with open(page, "w", encoding="utf-8") as f:
                f.write('<!doctype html><meta charset="utf-8"><body style="margin:0;background:#fff">'
                        '<span style="font-family:\'{}\';font-size:{}px;white-space:nowrap;padding:4px">{}</span>'
                        .format(font_name, size_px, html.escape(text)))
            subprocess.run([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=2400,120",
                            "--screenshot=" + png, "file:///" + page.replace(os.sep, "/")],
                           check=True, timeout=120, capture_output=True)
            g = np.asarray(Image.open(png).convert("L"), dtype=np.uint8)
        ys, xs = np.where(g < 250)
        if not len(xs):
            return None
        crop = g[max(ys.min() - 4, 0):ys.max() + 5, max(xs.min() - 4, 0):xs.max() + 5]
        rgba = np.zeros(crop.shape + (4,), dtype=np.uint8)
        rgba[..., 3] = 255 - crop  # black text, alpha from darkness
        return rgba
    return render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["English", "Telugu"], required=True)
    ap.add_argument("--results", default=None, help="Results-layout folder (default Results/<lang>)")
    ap.add_argument("--browser", default=None, help="Chromium/Edge executable for shaped Telugu titles")
    a = ap.parse_args()
    base = a.results or os.path.join(ROOT, "Results", a.lang)
    font = plotting.find_font() if a.lang == "Telugu" else None
    if a.lang == "Telugu" and a.browser:
        plotting.shaped_text_image = browser_shaped_text_image(a.browser)
    for path in sorted(glob.glob(os.path.join(base, "variance_raw", "*.json"))):
        rec = json.load(open(path, encoding="utf-8"))
        plotting.plot_variances(rec, os.path.join(base, "variance_adaptor_plots"), font)
        print("plots:", rec["id"], plotting.TITLE_RENDERING["mode"])
    csv = os.path.join(base, "losses.csv")
    if os.path.exists(csv):
        for split, name in ((("train",), "training_loss.png"), (("val",), "validation_loss.png"),
                            (("train", "val"), "loss_curves.png")):
            title = "{}: {}".format(a.lang, {"training_loss.png": "training losses", "validation_loss.png": "validation losses",
                                              "loss_curves.png": "training vs validation losses"}[name])
            plotting.plot_losses([csv], os.path.join(base, name), title, split)
        print("loss plots written")


if __name__ == "__main__":
    main()
