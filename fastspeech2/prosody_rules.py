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
Questions ("?") are left as predicted by rules 1-2. Rule 3 (phrasing, optional) adds:
3. Phrase boundaries and sentence types (English). Listeners hear punctuation through pause length,
   pre-boundary lengthening and the boundary tone; the model, trained on trimmed clips with few
   internal pauses, gives short pauses and almost no lengthening (blind listening: commas and "Wait!"
   were not heard). Before every internal pause the last syllable's rhyme is lengthened (x PRE_BOUNDARY)
   and the pause is raised to a minimum (PAUSE_MIN_FRAMES, by punctuation). A clause ending in "!" gets
   emphasis on its stressed vowel; a yes/no question (clause starting with an auxiliary) ends in a rise
   (final vowels raised to RISE_RATIO x the median), a wh-question gets the statement fall. Clarity
   micro-pauses (clarity.py, token "<pause>") are set to exactly their target length.
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

PRE_BOUNDARY = 1.25          # rhyme of the last syllable before an internal pause
PAUSE_MIN_FRAMES = {",": 14, ";": 20, ":": 20, "-": 14, "!": 28, "?": 28, ".": 28}  # 11.6 ms frames
EMPHASIS_PITCH, EMPHASIS_ENERGY = 1.12, 1.2
RISE_RATIO = 1.3             # last vowel of a yes/no question, x median vowel pitch
AUXILIARIES = set("am is are was were do does did have has had can could will would shall should may might "
                  "must isn't aren't wasn't weren't don't doesn't didn't haven't hasn't can't couldn't won't "
                  "wouldn't shouldn't".split())
WH_WORDS = {"what", "why", "who", "whom", "whose", "where", "when", "which", "how"}


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


def pauses_and_clauses(phones, words):
    """[(pause phone index or None, punctuation token, [word spans of the clause])] for each clause of the
    text, in order; the last clause has pause index None when the text ends without a pause phone."""
    i, clauses, cur = 0, [], []
    for w in words:
        wp = w["phones"]
        if not wp:
            continue
        if all(p in SILENCES for p in wp):
            if i < len(phones) and phones[i] in SILENCES:
                clauses.append((i, w["token"], cur))
                if w["token"] != "<pause>":
                    cur = []
                i += 1
            continue
        while phones[i:i + len(wp)] != wp and i < len(phones) and phones[i] in SILENCES:
            i += 1
        cur = cur + [(w["token"], list(range(i, i + len(wp))))]
        i += len(wp)
    clauses.append((None, None, cur))
    return clauses


def phrasing_factors(phones, words, text, pitch, dur, lang, median):
    """Rule 3. Returns (pf, ef, df, log); df entries for pauses are absolute targets / predicted."""
    n = len(phones)
    pf, ef, df = np.ones(n), np.ones(n), np.ones(n)
    log = []
    if lang != "en":
        return pf, ef, df, log
    final = text.rstrip()[-1:] if text.strip() else ""
    clauses = pauses_and_clauses(phones, words)
    for k, (pi, punct, spans) in enumerate(clauses):
        if pi is not None and punct == "<pause>":
            df[pi] = MICRO_PAUSE_FRAMES / max(dur[pi], 1.0)
            log.append({"rule": "micro_pause", "phone": pi, "frames": MICRO_PAUSE_FRAMES})
            continue
        if not spans:
            continue
        last = spans[-1][1]
        if pi is not None:  # internal boundary
            v = [j for j in last if is_vowel(phones[j], lang)]
            if v:
                for j in range(v[-1], last[-1] + 1):
                    df[j] *= PRE_BOUNDARY
            target = PAUSE_MIN_FRAMES.get(punct, 14)
            if dur[pi] < target:
                df[pi] = target / max(dur[pi], 1.0)
            log.append({"rule": "boundary", "punct": punct, "pause_phone": pi, "pause_frames": round(float(dur[pi] * df[pi]))})
        punct_k = punct if pi is not None else final
        if punct_k == "!":
            v = [j for j in last if is_vowel(phones[j], lang)]
            s1 = [j for j in v if phones[j].endswith("1")] or v[-1:]
            for j in s1[:1]:
                pf[j] *= EMPHASIS_PITCH
                ef[j] *= EMPHASIS_ENERGY
                log.append({"rule": "exclamation_emphasis", "phone": j})
        if punct_k == "?" and spans:
            first = spans[0][0].lower()
            if first in AUXILIARIES:
                v = [j for j in last if is_vowel(phones[j], lang)]
                s1 = [j for j in v if phones[j].endswith("1")]
                rise = [j for j in v if not s1 or j >= s1[-1]]
                for m, j in enumerate(rise):
                    target = median * (1 + (RISE_RATIO - 1) * (m + 1) / len(rise))
                    if pitch[j] < target:
                        pf[j] *= target / pitch[j]
                        log.append({"rule": "question_rise", "phone": j, "from_hz": round(float(pitch[j]), 1),
                                    "to_hz": round(float(target), 1)})
    return pf, ef, df, log


MICRO_PAUSE_FRAMES = 6


def factors(phones, words, text, pitch_hz, energy, dur_frames, lang, fall=True, stress=1.0, phrasing=False):
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
    ends = text.rstrip().endswith((".", "!"))
    if phrasing and text.rstrip().endswith("?"):  # wh-question: statement fall
        last_clause = pauses_and_clauses(phones, words)[-1][2]
        ends = bool(last_clause) and last_clause[0][0].lower() in WH_WORDS
    if fall and ends:
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
    if phrasing:
        p3, e3, d3, log3 = phrasing_factors(phones, words, text, pitch * pf, dur * df, lang, median)
        pf, ef, log = pf * p3, ef * e3, log + log3
        pz = np.array([p in SILENCES for p in phones])
        df = np.where(pz & (d3 != 1), d3, df * d3)  # pause targets are absolute
    return pf, ef, df, log


def second_pass_controls(model, output, phones, words, text, lang, fall, stress, base, phrasing=False):
    """Per-phone control tensors for the second pass, from the first-pass model output.
    base: (p_control, e_control, d_control) of the first pass (floats, or a (1, L) tensor for d_control)."""
    import torch
    va = model.variance_adaptor
    n = len(phones)
    pitch = output[2][0, :n].detach().cpu().numpy() * va.pitch_std + va.pitch_mean
    energy = output[3][0, :n].detach().cpu().numpy() * va.energy_std + va.energy_mean
    dur = output[5][0, :n].detach().cpu().numpy()
    pf, ef, df, log = factors(phones, words, text, pitch / base[0], energy / base[1], dur, lang, fall=fall, stress=stress,
                              phrasing=phrasing)
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
