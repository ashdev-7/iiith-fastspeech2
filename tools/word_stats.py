"""Per-word duration/pitch statistics of the test sentences (the numbers behind the report's Section 7 analysis).

Reads Results/<Lang>/variance_raw/<ID>.json and frontend_trace.json. Word duration = sum of its phones;
word pitch = max phone F0 (predicted, Hz). Also: mean vowel duration, final-vowel lengthening, pauses,
declination (mean F0 of the first vs last third of the phones), speaking rate, and for "conduct" the
duration and pitch per syllable.

--source model uses the model's own predictions (no inference-time controls) where they were saved.

Usage: python tools/word_stats.py [--source final|model] [--json out.json]     (pure Python, no third-party packages)
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
SIL = {"sp", "spn", "sil"}
TE_VOWELS = {"a", "aa", "i", "ii", "u", "uu", "e", "ee", "o", "oo", "ai", "au", "rq", "rqq"}


def is_vowel(p, lang):
    return p[-1] in "012" if lang == "en" else p in TE_VOWELS


def words_of(raw, trace):
    """[(token, [phone indices])], pauses as ("<pause>", [i])."""
    phones, out, i = raw["phones"], [], 0
    for w in trace["words"]:
        n = len(w["phones"])
        if n and all(p in SIL for p in w["phones"]):
            if i < len(phones) and phones[i] in SIL:
                out.append(("<pause>", [i]))
                i += 1
            continue
        while phones[i:i + n] != w["phones"] and i < len(phones) and phones[i] in SIL:
            out.append(("<pause>", [i]))
            i += 1
        assert phones[i:i + n] == w["phones"], (w["token"], phones[i:i + n], w["phones"])
        out.append((w["token"], list(range(i, i + n))))
        i += n
    return out


FRAME_MS = 256 / 22050 * 1000


def sentence_stats(raw, tr, lang, d, f0):
    """Statistics of one sentence for per-phone durations d (ms) and phone F0 f0 (Hz)."""
    ph = raw["phones"]
    speech = [k for k, p in enumerate(ph) if p not in SIL]
    vowels = [k for k in speech if is_vowel(ph[k], lang)]
    vmean = sum(d[k] for k in vowels) / len(vowels)
    last_v = vowels[-1]
    words = []
    for tok, idx in words_of(raw, tr):
        if tok == "<pause>":
            words.append({"word": "<pause>", "ms": round(d[idx[0]])})
            continue
        words.append({"word": tok, "ms": round(sum(d[k] for k in idx)), "peak_hz": round(max(f0[k] for k in idx))})
    third = max(1, len(speech) // 3)
    r = {
        "total_s": round(sum(d) / 1000, 2),
        "vowel_mean_ms": round(vmean), "final_vowel_ms": round(d[last_v]),
        "final_lengthening": round(d[last_v] / vmean, 1),
        "pauses_ms": [w["ms"] for w in words if w["word"] == "<pause>"],
        "first_third_hz": round(sum(f0[k] for k in speech[:third]) / third),
        "last_third_hz": round(sum(f0[k] for k in speech[-third:]) / third),
        "phones_per_s": round(len(speech) / (sum(d[k] for k in speech) / 1000), 1),
        "words": words,
    }
    content = [w for w in words if w["word"] != "<pause>"]
    r["peak"] = max(content, key=lambda w: w["peak_hz"])
    r["lowest"] = min(content, key=lambda w: w["peak_hz"])
    for tok, idx in words_of(raw, tr):
        if tok.lower() == "conduct":  # K AA N | D AH K T
            s1, s2 = idx[:3], idx[3:]
            r["conduct"] = {"phones": " ".join(ph[k] for k in idx),
                            "syll1_ms": round(sum(d[k] for k in s1)), "syll2_ms": round(sum(d[k] for k in s2)),
                            "ratio": round(sum(d[k] for k in s1) / sum(d[k] for k in s2), 2),
                            "v1_hz": round(f0[idx[1]]), "v2_hz": round(f0[idx[4]]),
                            "word_ms": round(sum(d[k] for k in idx)),
                            "range_hz": [round(min(f0[k] for k in idx)), round(max(f0[k] for k in idx))]}
    return r


def stats(lang_dir, source="final"):
    """source "final": the values used for the audio; "model": the model's own predictions for the same phones
    (variance_raw[...]["model_prediction"], written when inference-time controls are active; otherwise final)."""
    lang = "en" if lang_dir == "English" else "te"
    base = os.path.join(ROOT, "Results", lang_dir)
    trace = json.load(open(os.path.join(base, "frontend_trace.json"), encoding="utf-8"))
    res = {}
    for sid, tr in trace.items():
        raw = json.load(open(os.path.join(base, "variance_raw", sid + ".json"), encoding="utf-8"))
        mp = raw.get("model_prediction") if source == "model" else None
        if mp:
            d, f0 = [x * FRAME_MS for x in mp["duration_frames"]], mp["pitch_hz"]
        else:
            d, f0 = raw["duration_ms"], raw["pitch_hz"]
        res[sid] = sentence_stats(raw, tr, lang, d, f0)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--source", choices=["final", "model"], default="final")
    a = ap.parse_args()
    out = {lang: stats(lang, a.source) for lang in ("English", "Telugu")}
    for lang, rs in out.items():
        for sid, r in rs.items():
            print("{}: {} s, vowel mean {} ms, final vowel {} ms (x{}); F0 first third {} -> last third {} Hz; "
                  "peak {} {} Hz, lowest {} {} Hz; pauses {}; {} phones/s".format(
                      sid, r["total_s"], r["vowel_mean_ms"], r["final_vowel_ms"], r["final_lengthening"],
                      r["first_third_hz"], r["last_third_hz"], r["peak"]["word"], r["peak"]["peak_hz"],
                      r["lowest"]["word"], r["lowest"]["peak_hz"], r["pauses_ms"], r["phones_per_s"]))
            print("    " + ", ".join("{}({}{})".format(w["word"], w["ms"], "," + str(w["peak_hz"]) if "peak_hz" in w else "")
                                     for w in r["words"]))
            if "conduct" in r:
                print("    conduct:", r["conduct"])
    if a.json:
        json.dump(out, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
