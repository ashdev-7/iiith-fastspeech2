"""Praat-based acoustic analysis of generated speech, per sentence and per word.

For every test sentence (Results/<Lang>/test_generated_audios) it maps the predicted phone durations
(variance_raw/<ID>.json) onto the words of frontend_trace.json and measures, per word and per sentence:
  duration, phones/s, intensity (dB), HNR (harmonics-to-noise, dB; low = breathy/noisy/"muffled"),
  spectral centre of gravity (Hz; low = dull/muffled), high-band energy share (4-11 kHz, dB re total;
  low = missing fricative/burst detail), F0 median/range, voiced fraction.
Words are flagged when they fall well below the sentence's own median on HNR or high-band share, or
are very fast (phones/s), which is how "muffled" or "rushed" words show up acoustically.

Optional --ref_dir: a folder of natural recordings (wav) of the same voice; their sentence-level
values are printed as the baseline the synthesis should approach.

Usage: python tools/acoustic_analysis.py --lang English [--ref_dir DIR] [--json out.json]
Needs numpy and praat-parselmouth.
"""
import argparse
import glob
import json
import os
import statistics

import numpy as np
import parselmouth
from parselmouth.praat import call

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SILENCES = {"sp", "spn", "sil", "@sp", "@spn", "@sil"}


def band_db(seg, lo, hi):
    spec = seg.to_spectrum()
    tot = spec.get_band_energy(0, seg.sampling_frequency / 2) + 1e-20
    return 10 * np.log10(spec.get_band_energy(lo, hi) / tot + 1e-20), spec.get_centre_of_gravity(2.0)


class Tracks:
    """Frame tracks computed once on the whole sound (words are often shorter than Praat's windows)."""

    def __init__(self, snd):
        inten = snd.to_intensity(minimum_pitch=75.0, time_step=0.005, subtract_mean=False)
        self.i_t, self.i_v = inten.xs(), inten.values[0]
        hnr = snd.to_harmonicity_cc(time_step=0.005, minimum_pitch=75.0)
        self.h_t, self.h_v = hnr.xs(), hnr.values[0]
        pitch = snd.to_pitch(time_step=0.005, pitch_floor=60.0, pitch_ceiling=450.0)
        self.p_t, self.p_v = pitch.xs(), pitch.selected_array["frequency"]

    @staticmethod
    def sel(t, v, t0, t1):
        return v[(t >= t0) & (t < t1)]


def measure(snd, tracks, t0=None, t1=None):
    seg = snd if t0 is None else snd.extract_part(from_time=t0, to_time=t1, preserve_times=False)
    t0 = 0.0 if t0 is None else t0
    t1 = snd.get_total_duration() if t1 is None else t1
    dur = seg.get_total_duration()
    out = {"dur_ms": round(dur * 1000, 1)}
    if dur < 0.02:
        return out
    iv = Tracks.sel(tracks.i_t, tracks.i_v, t0, t1)
    out["intensity_db"] = round(float(10 * np.log10(np.mean(10 ** (iv / 10)))), 1) if iv.size else None
    hv = Tracks.sel(tracks.h_t, tracks.h_v, t0, t1)
    hv = hv[hv > -100]
    out["hnr_db"] = round(float(np.mean(hv)), 1) if hv.size else None
    hf, cog = band_db(seg, 4000, seg.sampling_frequency / 2)
    out["hf_share_db"] = round(float(hf), 1)
    out["cog_hz"] = round(float(cog))
    f0 = Tracks.sel(tracks.p_t, tracks.p_v, t0, t1)
    f0v = f0[f0 > 0]
    out["voiced_frac"] = round(float(f0v.size / max(f0.size, 1)), 2)
    if f0v.size:
        out["f0_med"] = round(float(np.median(f0v)))
        out["f0_range_st"] = round(float(12 * np.log2(np.percentile(f0v, 95) / np.percentile(f0v, 5))), 1)
    return out


def sentence_level(snd):
    m = measure(snd, Tracks(snd))
    pp = call(snd.to_pitch(time_step=0.005, pitch_floor=60.0, pitch_ceiling=450.0), "To PointProcess")
    m["jitter_pct"] = round(100 * call(pp, "Get jitter (local)", 0, 0, 1e-4, 0.02, 1.3), 2)
    m["shimmer_pct"] = round(100 * call([snd, pp], "Get shimmer (local)", 0, 0, 1e-4, 0.02, 1.3, 1.6), 2)
    return m


def words_with_times(raw, trace):
    phones, dur = raw["phones"], [d / 1000 for d in raw["duration_ms"]]
    starts = np.concatenate([[0], np.cumsum(dur)])
    i, out = 0, []
    for w in trace["words"]:
        n = len(w["phones"])
        if w["phones"] and all(p in SILENCES for p in w["phones"]):
            # punctuation pause: consumed only if present in the phone sequence
            if i < len(phones) and phones[i] in SILENCES:
                out.append({"word": w["token"], "pause": True, "t0": starts[i], "t1": starts[i + 1]})
                i += 1
            continue
        if phones[i:i + n] != w["phones"]:
            # skip stray silences the frontend inserted (e.g. word-internal space)
            while i < len(phones) and phones[i] in SILENCES:
                i += 1
            if phones[i:i + n] != w["phones"]:
                raise ValueError("phone mismatch at word {!r}: {} vs {}".format(w["token"], phones[i:i + n], w["phones"]))
        out.append({"word": w["token"], "n_phones": n, "phones": " ".join(w["phones"]), "t0": starts[i], "t1": starts[i + n]})
        i += n
    return out


def analyse_lang(lang):
    base = os.path.join(ROOT, "Results", lang)
    trace = json.load(open(os.path.join(base, "frontend_trace.json"), encoding="utf-8"))
    report = {}
    for sid, tr in trace.items():
        wav = glob.glob(os.path.join(base, "test_generated_audios", sid + "_*.wav"))[0]
        raw = json.load(open(os.path.join(base, "variance_raw", sid + ".json"), encoding="utf-8"))
        snd = parselmouth.Sound(wav)
        sent = sentence_level(snd)
        tracks = Tracks(snd)
        words = []
        for w in words_with_times(raw, tr):
            t1 = min(w["t1"], snd.get_total_duration())
            m = measure(snd, tracks, w["t0"], t1)
            m.update({k: w[k] for k in ("word", "n_phones", "phones", "pause") if k in w})
            if "n_phones" in w:
                m["phones_per_s"] = round(w["n_phones"] / max(t1 - w["t0"], 1e-3), 1)
            words.append(m)
        content = [w for w in words if not w.get("pause") and w.get("hnr_db") is not None and w.get("intensity_db") is not None]
        med = {k: statistics.median(w[k] for w in content) for k in ("hnr_db", "hf_share_db", "phones_per_s", "intensity_db")}
        for w in content:
            flags = []
            if w["hnr_db"] < med["hnr_db"] - 4:
                flags.append("noisy/breathy (HNR {:.1f} vs {:.1f})".format(w["hnr_db"], med["hnr_db"]))
            if w["hf_share_db"] < med["hf_share_db"] - 6:
                flags.append("dull: little energy above 4 kHz ({:.1f} vs {:.1f} dB)".format(w["hf_share_db"], med["hf_share_db"]))
            if w["phones_per_s"] > med["phones_per_s"] * 1.5:
                flags.append("rushed ({:.0f} vs {:.0f} phones/s)".format(w["phones_per_s"], med["phones_per_s"]))
            if w["intensity_db"] < med["intensity_db"] - 6:
                flags.append("weak ({:.0f} vs {:.0f} dB)".format(w["intensity_db"], med["intensity_db"]))
            w["flags"] = flags
        report[sid] = {"sentence": sent, "word_medians": med, "words": words}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["English", "Telugu"])
    ap.add_argument("--ref_dir")
    ap.add_argument("--json")
    a = ap.parse_args()
    rep = analyse_lang(a.lang)
    for sid, r in rep.items():
        s = r["sentence"]
        print("\n== {}  {}".format(sid, "  ".join("{}={}".format(k, v) for k, v in s.items())))
        for w in r["words"]:
            if w.get("pause"):
                print("   [pause {:.0f} ms]".format(w["dur_ms"]))
                continue
            print("   {:<30} {:>6.0f} ms {:>5} ph/s HNR {:>5} HF {:>6} CoG {:>5} {}".format(
                w["word"][:30], w["dur_ms"], w.get("phones_per_s"), w.get("hnr_db"), w.get("hf_share_db"),
                w.get("cog_hz"), "; ".join(w.get("flags", []))))
    if a.ref_dir:
        refs = [sentence_level(parselmouth.Sound(f)) for f in sorted(glob.glob(os.path.join(a.ref_dir, "*.wav")))]
        keys = ["hnr_db", "hf_share_db", "cog_hz", "jitter_pct", "shimmer_pct", "f0_range_st"]
        print("\nreference recordings (n={}): ".format(len(refs))
              + "  ".join("{}={:.2f}".format(k, statistics.median(r[k] for r in refs if r.get(k) is not None)) for k in keys))
    if a.json:
        json.dump(rep, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
