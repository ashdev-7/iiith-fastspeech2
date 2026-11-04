"""Inference-time prosody rules on the variance adaptor's own predictions (no retraining).

FastSpeech 2 exposes duration, pitch and energy per phone; these rules read the model's first-pass
predictions and return per-phone factors for a second pass (the same controls as --pitch_control etc.,
applied per phone). Two general rules, each triggered only when the prediction violates it:

1. Lexical stress (English). In a word with a primary-stressed vowel (digit 1) and unstressed vowels
   (digit 0), the stressed vowel should be the most prominent. Measured on the test sentences: the model
   realises this on only 11 of 28 such words (report 5.1), e.g. "conduct" (noun) with the louder, longer,
   higher vowel on the second syllable. Where violated, the stressed vowel's pitch, duration and energy
   are raised towards a margin over the strongest unstressed vowel of the word (capped), and unstressed
   vowels longer than it are shortened (capped).
2. Terminal fall (both languages). In a statement (text ends in "." or "!"), the vowels after the
   nuclear stress of the last word should be low. Measured: nine of the ten test sentences end at
   0.74-0.89 x the sentence's median vowel pitch; E3 ends at 1.11 x (a continuation rise on "-ly").
   Where violated, those vowels are lowered to FALL_RATIO x the median.
Questions ("?") are left as predicted.
"""
import numpy as np

SILENCES = {"sp", "spn", "sil", ""}
EN_VOWELS = set("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split())
TE_VOWELS = set("a aa i ii u uu e ee ai o oo au".split())  # text/telugu/phones.py

FALL_RATIO = 0.85        # post-nuclear vowels of a statement: at most 0.85 x median vowel pitch
PITCH_MARGIN = 1.04      # stressed vowel >= 1.04 x the highest unstressed vowel of the word (~0.7 semitone)
PITCH_CAP = 1.12
DUR_MARGIN = 1.15        # stressed vowel >= 1.15 x the longest unstressed vowel
DUR_CAP, DUR_FLOOR = 1.4, 0.75
ENERGY_MARGIN = 1.10
ENERGY_CAP = 1.5


def is_vowel(phone, lang):
    if lang == "en":
        return phone.rstrip("012") in EN_VOWELS
    return phone in TE_VOWELS


def align_words(phones, words):
    """[(token, [phone indices])] for the frontend trace's words; punctuation pauses are skipped."""
    i, out = 0, []
    for w in words:
        wp = w["phones"]
        if not wp:
            continue
        if all(p in SILENCES for p in wp):
            if i < len(phones) and phones[i] in SILENCES:
                i += 1
            continue
        while phones[i:i + len(wp)] != wp and i < len(phones) and phones[i] in SILENCES:
            i += 1
        if phones[i:i + len(wp)] != wp:
            raise ValueError("phone mismatch at word {!r}".format(w["token"]))
        out.append((w["token"], list(range(i, i + len(wp)))))
        i += len(wp)
    return out


def factors(phones, words, text, pitch_hz, energy, dur_frames, lang, fall=True, stress=1.0):
    """Per-phone (pitch, energy, duration) factors and a log of what fired.
    fall: apply rule 2. stress: strength of rule 1 in [0, 1]; each factor f becomes 1 + stress * (f - 1)."""
    n = len(phones)
    pf, ef, df = np.ones(n), np.ones(n), np.ones(n)
    pitch = np.asarray(pitch_hz, dtype=float).copy()
    energy = np.asarray(energy, dtype=float)
    dur = np.maximum(np.asarray(dur_frames, dtype=float), 1.0)
    log = []
    aligned = align_words(phones, words)
    vowels = [j for _, idx in aligned for j in idx if is_vowel(phones[j], lang)]
    if not vowels:
        return pf, ef, df, log
    median = float(np.median(pitch[vowels]))

    # Rule 2 first, so that rule 1 compares against the corrected final vowels.
    if fall and text.rstrip().endswith((".", "!")):
        last = next((idx for _, idx in reversed(aligned) if any(is_vowel(phones[j], lang) for j in idx)), None)
        lw = [j for j in last if is_vowel(phones[j], lang)]
        stressed = [j for j in lw if lang == "en" and phones[j].endswith("1")]
        post = [j for j in lw if j > stressed[-1]] if stressed else lw[-1:]
        for j in post:
            target = FALL_RATIO * median
            if pitch[j] > target:
                pf[j] = target / pitch[j]
                log.append({"rule": "terminal_fall", "phone": j, "from_hz": round(float(pitch[j]), 1),
                            "to_hz": round(float(target), 1)})
                pitch[j] = target

    if lang == "en" and stress > 0:
        sp, se, sd = np.ones(n), np.ones(n), np.ones(n)
        for token, idx in aligned:
            v = [j for j in idx if is_vowel(phones[j], lang)]
            s1 = [j for j in v if phones[j].endswith("1")]
            s0 = [j for j in v if phones[j].endswith("0")]
            if not s1 or not s0:
                continue
            s = s1[0]
            fired = {}
            up = PITCH_MARGIN * pitch[s0].max() / pitch[s]
            if up > 1:
                sp[s] = min(up, PITCH_CAP)
                fired["pitch"] = round(float(sp[s]), 3)
            up = ENERGY_MARGIN * energy[s0].max() / max(energy[s], 1e-6)
            if up > 1:
                se[s] = min(up, ENERGY_CAP)
                fired["energy"] = round(float(se[s]), 3)
            need = DUR_MARGIN * dur[s0].max() / dur[s]
            if need > 1:
                sd[s] = min(np.sqrt(need), DUR_CAP)
                for j in s0:
                    if dur[j] * DUR_MARGIN > dur[s] * sd[s]:
                        sd[j] = max(dur[s] * sd[s] / (DUR_MARGIN * dur[j]), DUR_FLOOR)
                fired["duration"] = round(float(sd[s]), 3)
            if fired:
                log.append({"rule": "lexical_stress", "word": token, "phone": s, "strength": stress, **fired})
        pf *= 1 + stress * (sp - 1)
        ef *= 1 + stress * (se - 1)
        df *= 1 + stress * (sd - 1)
    return pf, ef, df, log


def second_pass_controls(model, output, phones, words, text, lang, fall, stress, base):
    """Per-phone control tensors for the second pass, from the first-pass model output.
    base: (p_control, e_control, d_control) of the first pass (floats, or a (1, L) tensor for d_control)."""
    import torch
    va = model.variance_adaptor
    n = len(phones)
    pitch = output[2][0, :n].detach().cpu().numpy() * va.pitch_std + va.pitch_mean
    energy = output[3][0, :n].detach().cpu().numpy() * va.energy_std + va.energy_mean
    dur = output[5][0, :n].detach().cpu().numpy()
    pf, ef, df, log = factors(phones, words, text, pitch / base[0], energy / base[1], dur, lang, fall=fall, stress=stress)
    dev = output[2].device
    t = lambda a: torch.tensor([a], device=dev, dtype=torch.float32)  # noqa: E731
    return t(pf) * base[0], t(ef) * base[1], base[2] * t(df), log


def stress_realised(phones, words, pitch, energy, dur):
    """Words with a primary-stressed and an unstressed vowel, and how many of them give the stressed vowel
    more pitch, duration and energy than every unstressed vowel of the word on at least two of the three.
    Any monotonic scale works (Hz or z-score). Returns (n_words, n_realised, per-cue counts); per-cue counts
    include "chance", the expected number realised if each cue were random (p = 1/k for k candidate vowels)."""
    pitch, energy, dur = (np.asarray(a, dtype=float) for a in (pitch, energy, dur))
    n = ok = 0
    cues = {"pitch": 0, "duration": 0, "energy": 0, "chance": 0.0}
    for _, idx in align_words(phones, words):
        v = [j for j in idx if is_vowel(phones[j], "en")]
        s1 = [j for j in v if phones[j].endswith("1")]
        s0 = [j for j in v if phones[j].endswith("0")]
        if not s1 or not s0:
            continue
        s = s1[0]
        q = 1.0 / (len(s0) + 1)
        cues["chance"] += 3 * q * q * (1 - q) + q ** 3
        c = {"pitch": pitch[s] > pitch[s0].max(), "duration": dur[s] > dur[s0].max(), "energy": energy[s] > energy[s0].max()}
        for k in c:
            cues[k] += bool(c[k])
        n += 1
        ok += sum(c.values()) >= 2
    return n, ok, cues
