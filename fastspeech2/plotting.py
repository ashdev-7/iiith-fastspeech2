"""Variance-adaptor plots (plan.txt section 7) and loss curves.

Each variance plot is drawn from the saved variance_raw/<ID>.json, so it can be regenerated without
the model. Pitch and energy are phone-level predictions, drawn as steps over each phone's predicted
time span. The pitch plot also overlays the F0 that pyworld extracts from the generated wav
(unvoiced frames left as gaps), labelled as such; no ground truth exists for the test sentences.
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TELUGU_FONTS = ("Noto Sans Telugu", "Nirmala UI", "Gautami", "Lohit Telugu", "Pothana2000")


def find_font(names=TELUGU_FONTS, extra_dirs=()):
    for d in extra_dirs:
        for f in font_manager.findSystemFonts(fontpaths=[d]):
            font_manager.fontManager.addfont(f)
    available = {f.name for f in font_manager.fontManager.ttflist}
    for n in names:
        if n in available:
            return n
    return None


def _is_telugu(text):
    return any("ఀ" <= ch <= "౿" for ch in text)


def shaped_text_image(text, font_name, size_px=30):
    """Render text with Pillow's raqm layout (HarfBuzz shaping: Telugu conjuncts and vowel signs
    are composed correctly; matplotlib's own text engine does not shape Indic scripts).
    Returns an RGBA array, or None when raqm or the font file is unavailable."""
    try:
        from PIL import Image, ImageDraw, ImageFont, features
        if not features.check("raqm"):
            return None
        path = font_manager.findfont(font_manager.FontProperties(family=font_name), fallback_to_default=False)
        font = ImageFont.truetype(path, size_px, layout_engine=ImageFont.Layout.RAQM)
        left, top, right, bottom = font.getbbox(text)
        img = Image.new("RGBA", (right - left + 8, bottom - top + 8), (255, 255, 255, 0))
        ImageDraw.Draw(img).text((4 - left, 4 - top), text, font=font, fill=(0, 0, 0, 255))
        return np.asarray(img)
    except Exception:  # noqa: BLE001  (any failure -> caller falls back to matplotlib text)
        return None


TITLE_RENDERING = {"mode": None}  # recorded in inference_report.json


def _set_title(ax, item_id, text, what, telugu_font):
    """Line 1 (Latin) in the default font; line 2 = the original sentence. Telugu sentences are
    drawn as a shaped image when possible, else as text with per-glyph font fallback."""
    if not _is_telugu(text):
        ax.set_title("{} - {}\n{}".format(item_id, what, text), fontsize=10)
        return
    if telugu_font is None:  # no Telugu font: never render tofu boxes
        TITLE_RENDERING["mode"] = "no Telugu font: sentence omitted from plot titles"
        ax.set_title("{} - {} (sentence text: see variance_raw/{}.json)".format(item_id, what, item_id), fontsize=10)
        return
    img = shaped_text_image(text, telugu_font)
    if img is not None:
        from matplotlib.offsetbox import AnnotationBbox, OffsetImage
        TITLE_RENDERING["mode"] = "Pillow raqm (shaped) + " + telugu_font
        ax.add_artist(AnnotationBbox(OffsetImage(img, zoom=0.5), (0.5, 1.01), xycoords="axes fraction",
                                     box_alignment=(0.5, 0.0), frameon=False))
        ax.set_title("{} - {}".format(item_id, what), fontsize=10, pad=8 + img.shape[0] * 0.5 * 0.72)
    else:
        TITLE_RENDERING["mode"] = "matplotlib text with font fallback (Indic shaping not available)"
        ax.set_title("{} - {}\n{}".format(item_id, what, text), fontsize=10,
                     fontfamily=["DejaVu Sans", telugu_font])


def _phone_axis(ax, phones, starts, durs, frame_ms):
    centers = (starts + durs / 2.0) * frame_ms
    ax.set_xticks(centers)
    ax.set_xticklabels(phones, rotation=90, fontsize=7)
    ax.set_xlim(0, (starts[-1] + durs[-1]) * frame_ms if len(durs) else 1)
    for b in starts * frame_ms:
        ax.axvline(b, color="0.85", lw=0.5, zorder=0)
    ax.set_xlabel("phone (positioned at its predicted time span, ms)")


def plot_variances(rec, out_dir, telugu_font=None):
    """rec: dict from variance_raw/<ID>.json. Writes duration/, pitch/, energy/ PNGs; returns paths."""
    item_id, text = rec["id"], rec["original_text"]
    phones = rec["phones"]
    durs = np.array(rec["duration_frames"], dtype=float)
    frame_ms = rec["frame_ms"]
    starts = np.concatenate([[0.0], np.cumsum(durs)[:-1]])
    paths = {}
    width = max(8.0, 0.22 * len(phones) + 2)

    # Duration: bar per phone (frames), secondary axis in ms.
    fig, ax = plt.subplots(figsize=(width, 4))
    x = np.arange(len(phones))
    ax.bar(x, durs, color="#3a6ea5")
    ax.set_xticks(x)
    ax.set_xticklabels(phones, rotation=90, fontsize=7)
    ax.set_xlim(-0.6, len(phones) - 0.4)
    ax.set_ylabel("predicted duration (frames)")
    ax.set_xlabel("phone (input order)")
    sec = ax.secondary_yaxis("right", functions=(lambda f: f * frame_ms, lambda m: m / frame_ms))
    sec.set_ylabel("ms")
    _set_title(ax, item_id, text, "predicted duration, total {:.0f} frames = {:.2f} s".format(
        durs.sum(), durs.sum() * frame_ms / 1000), telugu_font)
    paths["duration"] = _save(fig, out_dir, "duration", item_id)

    # Pitch: phone-level prediction in Hz as steps, plus F0 of the generated audio.
    fig, ax = plt.subplots(figsize=(width, 4))
    pitch = np.array(rec["pitch_hz"], dtype=float)
    _steps(ax, starts, durs, pitch, frame_ms, "predicted F0 per phone (Hz)", "#c0392b")
    if rec.get("generated_f0_hz") is not None:
        f0 = np.array(rec["generated_f0_hz"], dtype=float)
        f0[f0 <= 0] = np.nan
        t = np.arange(len(f0)) * rec["generated_f0_period_ms"]
        ax.plot(t, f0, ".", ms=2, color="0.35", label="F0 extracted from the generated wav (pyworld; unvoiced = gaps)")
    ax.set_ylabel("Hz")
    _phone_axis(ax, phones, starts, durs, frame_ms)
    ax.legend(fontsize=7, loc="upper right")
    _set_title(ax, item_id, text, "pitch", telugu_font)
    paths["pitch"] = _save(fig, out_dir, "pitch", item_id)

    # Energy: phone-level prediction (frame L2 norm of the STFT magnitude, arbitrary units).
    fig, ax = plt.subplots(figsize=(width, 4))
    energy = np.array(rec["energy"], dtype=float)
    _steps(ax, starts, durs, energy, frame_ms, "predicted energy per phone", "#27ae60")
    ax.set_ylabel("energy: frame L2 STFT magnitude (a.u.)")
    _phone_axis(ax, phones, starts, durs, frame_ms)
    ax.legend(fontsize=7, loc="upper right")
    _set_title(ax, item_id, text, "energy", telugu_font)
    paths["energy"] = _save(fig, out_dir, "energy", item_id)
    return paths


def _steps(ax, starts, durs, values, frame_ms, label, color):
    first = True
    for s, d, v in zip(starts, durs, values):
        if d <= 0:
            continue  # zero-duration phones occupy no time
        ax.hlines(v, s * frame_ms, (s + d) * frame_ms, colors=color, lw=2.5, label=label if first else None)
        first = False


def _save(fig, out_dir, kind, item_id):
    os.makedirs(os.path.join(out_dir, kind), exist_ok=True)
    path = os.path.join(out_dir, kind, "{}_{}.png".format(item_id, kind))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")  # "tight" also keeps the title image inside
    plt.close(fig)
    return path


def plot_losses(csv_paths, out_path, title):
    """losses.csv (from train.py; several resumed runs may be concatenated) -> train/val curves."""
    rows = []
    for p in csv_paths:
        with open(p, encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    # Keep the last record per (split, step) in case a resumed run repeated steps.
    last = {(r["split"], int(r["step"])): r for r in rows}
    names = ["total", "mel", "mel_postnet", "pitch", "energy", "duration"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for ax, name in zip(axes.flat, names):
        for split, color in (("train", "#3a6ea5"), ("val", "#c0392b")):
            pts = sorted((s, float(r[name])) for (sp, s), r in last.items() if sp == split)
            if pts:
                ax.plot(*zip(*pts), color=color, lw=1.2 if split == "train" else 1.8,
                        marker=None if split == "train" else "o", ms=2.5, label=split)
        ax.set_title(name + " loss")
        ax.set_yscale("log")
        from matplotlib.ticker import FuncFormatter, LogLocator
        fmt = FuncFormatter(lambda v, _: "{:g}".format(v))  # plain numbers instead of 3x10^0
        ax.yaxis.set_major_formatter(fmt)
        ax.yaxis.set_minor_locator(LogLocator(subs=(2, 3, 5)))
        ax.yaxis.set_minor_formatter(fmt)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("step")
    axes[0][0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
