"""English inference frontend: original text -> model input text -> ARPAbet phones (plan.txt 5.3).

The original text is never modified; every step is recorded in the returned trace.
Normalisation (model input only):
  * times "H:MM" (optionally followed by AM/PM) -> "ten forty five AM", AM/PM read as letter names (minutes 01-09 -> "oh five",
    00 -> "o'clock");
  * other numbers: the reference's inflect-based normalize_numbers (years 1000-2999 are read in
    pairs, e.g. 2026 -> "twenty twenty six"; days are cardinal, as written: "12 November" -> "twelve");
  * an hour followed by a.m./p.m. without minutes ("9 a.m.") -> "nine AM";
  * "%" -> "percent", "&" -> "and";
  * hyphens between letters -> space (inflect writes "forty-five"; the reference split it into a pause);
  * whitespace runs collapse to one space.
Pronunciation:
  * split words: a fragment that is not a lexicon word, followed by a fragment, whose concatenation
    IS a lexicon word, is pronounced as that word ("rhyth m" -> RHYTHM; first full run read it as
    "R AY1 DH" + the letter name "EH1 M"). Recorded in the trace as rule "joined split word";
  * homographs known to g2p_en (e.g. CONDUCT) -> g2p_en's POS rule over the whole sentence;
  * other words -> the first librispeech-lexicon.txt pronunciation (the lexicon MFA used);
  * out-of-vocabulary words -> g2p_en (CMUdict, then its neural model); a word-final "oo" spelling
    is /uː/ in English (bamboo, shampoo, kangaroo), so a final OW that g2p_en guesses for it becomes UW
    ("andaroo" -> ... R UW, not ... R OW);
  * acronyms (all-capital tokens): 2 letters are read as letter names (UK, US, AI); 3-6 letters use their
    CMUdict entry if it has one (NASA, BBC) and letter names otherwise (IIIT, FBI), with the AM/PM duration scale;
  * punctuation -> "sp", as in the reference.
"""
import re

from text.numbers import normalize_numbers

_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_month_day_re = re.compile(r"\b(" + _MONTHS + r")\s+([0-9]{1,2})\b(?![:.,]?[0-9])")
# Titles whose period is an abbreviation mark, not the end of a phrase: no pause after them.
ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "prof", "st", "jr", "sr", "vs", "mt"}
# Corrections to g2p_en's homograph table (pron_if_pos, other_pron, POS prefixes). g2p_en maps "read"
# tagged VBD to R IY1 D (inverted) and gives the verb "process" the rare sense pro-CESS ("to walk in a
# procession"); "process" therefore uses the lexicon (PRO-cess) for every part of speech.
HOMOGRAPH_OVERRIDES = {"read": (["R", "EH1", "D"], ["R", "IY1", "D"], ("VBD", "VBN"))}
NOT_HOMOGRAPHS = {"process"}
# "read" is the one common verb whose present and past are spelled alike but pronounced differently; the
# tagger cannot use time words ("I read the book yesterday" -> VBP). A non-infinitive "read" with a
# past-time word in the same clause (between punctuation marks) is therefore read as past (R EH1 D).
PAST_TIME_WORDS = {"yesterday", "ago", "last", "earlier", "previously", "once"}
_hour_ampm_re = re.compile(r"\b(1[0-2]|0?[1-9])\s*([AaPp])\.?\s*[Mm]\b\.?")
_time_re = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)(?:\s*([AaPp])\.?\s*[Mm]\b\.?)?")
_punct = set(",;.:!?-\"()[]")
_token_re = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|[^\sA-Za-z]")

# Uppercase AM/PM (as written in times, and as produced by the time rule) are read as letter names;
# the lexicon's first entry for AM is the verb "am" (AE1 M).
LETTER_WORDS = {"AM": ["EY1", "EH1", "M"], "PM": ["P", "IY1", "EH1", "M"]}
# Letter names are spoken slowly and separately ("A. M."); the model, trained on running words, gave
# AM 244 ms, so the two letters merged into one glide (Whisper heard "pm"). Their predicted durations
# are scaled by this factor at inference (per-phone duration control; trace "duration_scale").
LETTER_DURATION_SCALE = 1.6
LETTER_NAMES = {
    "A": ["EY1"], "B": ["B", "IY1"], "C": ["S", "IY1"], "D": ["D", "IY1"], "E": ["IY1"], "F": ["EH1", "F"],
    "G": ["JH", "IY1"], "H": ["EY1", "CH"], "I": ["AY1"], "J": ["JH", "EY1"], "K": ["K", "EY1"], "L": ["EH1", "L"],
    "M": ["EH1", "M"], "N": ["EH1", "N"], "O": ["OW1"], "P": ["P", "IY1"], "Q": ["K", "Y", "UW1"], "R": ["AA1", "R"],
    "S": ["EH1", "S"], "T": ["T", "IY1"], "U": ["Y", "UW1"], "V": ["V", "IY1"], "W": ["D", "AH1", "B", "AH0", "L", "Y", "UW0"],
    "X": ["EH1", "K", "S"], "Y": ["W", "AY1"], "Z": ["Z", "IY1"],
}

_lexicon = None
_g2p = None
VOWELS = set("AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split())


def _number_words(n):
    return normalize_numbers(str(n))


def _ordinal_suffix(n):
    return "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _expand_time(m):
    hour, minute, ampm = int(m.group(1)), int(m.group(2)), m.group(3)
    words = [_number_words(hour)]
    if minute == 0:
        words.append("o'clock")
    elif minute < 10:
        words += ["oh", _number_words(minute)]
    else:
        words.append(_number_words(minute))
    if ampm:
        words.append(ampm.upper() + "M")  # "AM"/"PM", read as letter names (see LETTER_WORDS)
    return " ".join(words)


def normalize(text):
    """Model-input text (case kept for POS tagging; lexicon lookup is case-insensitive)."""
    rules = []
    out = _time_re.sub(_expand_time, text)
    if out != text:
        rules.append("time")
    o1 = _month_day_re.sub(lambda m: "{} {}{}".format(m.group(1), m.group(2), _ordinal_suffix(int(m.group(2)))), out)
    if o1 != out:
        rules.append("month day ordinal")
    out = o1
    o2 = _hour_ampm_re.sub(lambda m: "{} {}M".format(_number_words(int(m.group(1))), m.group(2).upper()), out)
    if o2 != out:
        rules.append("hour a.m./p.m.")
    o3 = re.sub(r"\s*%", " percent", o2)
    o3 = re.sub(r"\s*&\s*", " and ", o3)
    if o3 != o2:
        rules.append("symbols")
    out = o3
    t2 = normalize_numbers(out)
    if t2 != out:
        rules.append("numbers")
    t3 = re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", " ", t2)
    if t3 != t2:
        rules.append("hyphen")
    t4 = re.sub(r"\s+", " ", t3).strip()
    if t4 != t3:
        rules.append("whitespace")
    return t4, rules


def join_split_words(tokens, lexicon):
    """Join word fragments for pronunciation: tok + next is a lexicon word and tok is not one
    ("rhyth", "m" -> "rhythm"). Returns (tokens, [rule descriptions])."""
    out, rules, i = [], [], 0
    while i < len(tokens):
        tok = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if (nxt is not None and tok[0].isalpha() and nxt[0].isalpha() and tok.lower() not in lexicon
                and (tok + nxt).lower() in lexicon):
            out.append(tok + nxt)
            rules.append("joined split word '{} {}' -> '{}'".format(tok, nxt, tok + nxt))
            i += 2
            continue
        out.append(tok)
        i += 1
    return out, rules


def fix_final_oo(word, pron):
    """English spelling: word-final "oo" is /uː/; replace a final OW guessed by g2p_en with UW."""
    if word.lower().endswith("oo") and pron and pron[-1].rstrip("012") == "OW":
        return pron[:-1] + ["UW" + pron[-1][2:]], True
    return pron, False


def _strip(p):
    return p.rstrip("012")


def careful_variant(variants):
    """The citation (careful-speech) form among a word's lexicon pronunciations.

    The reference takes the first listed pronunciation, which for several hundred words is a casual
    reduction (the lexicon lists both): a dropped T/D ("thousand" TH AW1 Z AH0 N, "twenty" T W EH1 N IY0,
    "parliamentary", "second"), a flapped T ("thirty" TH ER1 D IY0), g-dropping ("going" G OW1 IH0 N) or
    a leading HH before W ("one" HH W AH1 N). Read speech uses the full form, and the reduced forms are
    the ones listeners mishear. Returns (pronunciation, rule or None); any other difference keeps the first."""
    first = variants[0]
    for q in variants[1:]:
        if len(q) == len(first) + 1:
            for i in range(len(q)):
                if q[:i] + q[i + 1:] == first and q[i] in ("T", "D"):
                    return q, "restored dropped " + q[i]
        if len(q) == len(first):
            diff = [i for i in range(len(q)) if q[i] != first[i]]
            if len(diff) == 1 and first[-2:] == ["IH0", "N"] and diff[0] == len(q) - 1 and q[-1] == "NG":
                return q, "-ing (not -in)"
            core = [i for i in range(len(q)) if _strip(q[i]) != _strip(first[i])]
            if len(core) == 1 and first[core[0]] == "D" and q[core[0]] == "T" and all(
                    _strip(q[i]) not in ("D", "T") or i == core[0] for i in diff):
                return q, "unflapped T"
        if first[:2] == ["HH", "W"] and q == first[1:]:
            return q, "W (not HH W)"
    return first, None


def load_lexicon(path, careful=True):
    """word -> (pronunciation, careful-variant rule or None). careful=False: first pronunciation (reference)."""
    variants = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                variants.setdefault(parts[0].lower(), []).append(parts[1:])
    if not careful:
        return {w: (v[0], None) for w, v in variants.items()}
    return {w: careful_variant(v) for w, v in variants.items()}


COMPOUND_MIN_LETTERS, COMPOUND_MIN_PART = 12, 4


def compound_split(word, lexicon):
    """A long out-of-lexicon word as the fewest lexicon words, each >= COMPOUND_MIN_PART letters
    ("antidisestablishmentarianism" -> anti + disestablishment + arianism); None if impossible.
    Lexicon pronunciations of the parts are more reliable than g2p_en's guess for the whole word."""
    if len(word) < COMPOUND_MIN_LETTERS or not word.isalpha() or word in lexicon:
        return None
    best = {0: []}
    for i in range(len(word)):
        if i not in best:
            continue
        for j in range(i + COMPOUND_MIN_PART, len(word) + 1):
            if word[i:j] in lexicon and (j not in best or len(best[i]) + 1 < len(best[j])):
                best[j] = best[i] + [word[i:j]]
    parts = best.get(len(word))
    return parts if parts and len(parts) >= 2 else None


def _get_g2p():
    global _g2p
    if _g2p is None:
        import nltk
        # nltk >= 3.9 renamed the tagger resource; g2p_en only fetches the old names.
        for res in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger", "cmudict"):
            try:
                nltk.download(res, quiet=True)
            except Exception:
                pass
        from g2p_en import G2p
        _g2p = G2p()
    return _g2p


def text_to_phones(text, lexicon_path):
    """Returns (phones, trace). trace = {"original", "model_input", "rules", "words": [...]}."""
    global _lexicon
    if _lexicon is None:
        _lexicon = load_lexicon(lexicon_path)
    model_input, rules = normalize(text)
    tokens, joined = join_split_words(_token_re.findall(model_input), _lexicon)
    rules = rules + joined
    words = [t for t in tokens if t[0].isalpha()]
    g2p = _get_g2p()
    tags = dict()
    if any(w.lower() in g2p.homograph2features and w.lower() not in NOT_HOMOGRAPHS for w in words):
        from nltk import pos_tag
        tags = {i: tag for i, (_, tag) in enumerate(pos_tag(words))}

    clause_of, clause_words, c = {}, {}, 0  # word index -> clause number; clause -> its lower-case words
    for tok in tokens:
        if tok in _punct:
            c += 1
        elif tok[0].isalpha():
            clause_of[len(clause_of)] = c
            clause_words.setdefault(c, set()).add(tok.lower())
    phones, scales, word_trace, wi = [], [], [], 0
    prev = None
    for tok in tokens:
        after_abbrev = tok == "." and prev is not None and prev.lower() in ABBREVIATIONS
        prev = tok
        if after_abbrev:
            word_trace.append({"token": tok, "source": "abbreviation period (no pause)", "phones": []})
            continue
        if not tok[0].isalpha():
            if tok in _punct or not tok.strip():
                if phones and phones[-1] != "sp":
                    phones.append("sp")
                    scales.append(1.0)
                word_trace.append({"token": tok, "source": "punctuation", "phones": ["sp"]})
            else:
                word_trace.append({"token": tok, "source": "dropped (unsupported symbol)", "phones": []})
            continue
        low = tok.lower()
        if tok in LETTER_WORDS:
            pron, source = LETTER_WORDS[tok], "letter names"
        elif tok.isupper() and len(tok) == 2 or (tok.isupper() and 3 <= len(tok) <= 6 and low not in g2p.cmu):
            pron, source = [p for ch in tok for p in LETTER_NAMES[ch]], "letter names"
        elif tok.isupper() and 3 <= len(tok) <= 6:
            pron, source = g2p.cmu[low][0], "CMUdict (acronym)"
        elif low in HOMOGRAPH_OVERRIDES:
            pron1, pron2, pos1 = HOMOGRAPH_OVERRIDES[low]
            past = tags[wi].startswith(pos1) or (
                low == "read" and tags[wi] != "VB" and clause_words[clause_of[wi]] & PAST_TIME_WORDS)
            pron = pron1 if past else pron2
            source = "homograph override (POS {}{})".format(
                tags[wi], ", past-time word in clause" if past and not tags[wi].startswith(pos1) else "")
        elif low in g2p.homograph2features and low not in NOT_HOMOGRAPHS:
            pron1, pron2, pos1 = g2p.homograph2features[low]
            pron = pron1 if tags[wi].startswith(pos1) else pron2
            source = "g2p_en homograph (POS {})".format(tags[wi])
        elif low in _lexicon:
            pron, careful = _lexicon[low]
            source = "lexicon" + (" (careful variant: {})".format(careful) if careful else "")
        elif compound_split(low, _lexicon):
            parts = compound_split(low, _lexicon)
            pron = [p for part in parts for p in _lexicon[part][0]]
            source = "compound of lexicon words: " + " + ".join(parts)
        else:
            pron = [p for p in g2p(tok) if re.match(r"^[A-Z]+[012]?$", p)]
            source = "g2p_en (out of lexicon)"
            pron, fixed = fix_final_oo(tok, pron)
            if fixed:
                source += ", final 'oo' -> UW"
        start = len(phones)
        phones.extend(pron)
        scale = LETTER_DURATION_SCALE if source == "letter names" else 1.0
        scales.extend([scale] * len(pron))
        word_trace.append({"token": tok, "source": source, "phones": list(pron), "_start": start})
        if scale != 1.0:
            word_trace[-1]["duration_scale"] = scale
        wi += 1
    # "the" before a vowel is DH IY0 ("the orchestra", "the international"); DH AH0 + vowel is a hiatus
    # the model blurs into one syllable.
    for k, w in enumerate(word_trace):
        nxt = word_trace[k + 1] if k + 1 < len(word_trace) else None
        if (w["token"].lower() == "the" and w["phones"] == ["DH", "AH0"] and nxt is not None and "_start" in nxt
                and nxt["phones"] and _strip(nxt["phones"][0]) in VOWELS):
            w["phones"] = ["DH", "IY0"]
            phones[w["_start"] + 1] = "IY0"
            w["source"] += " ('the' before a vowel)"
    for w in word_trace:
        w.pop("_start", None)
    while phones and phones[-1] == "sp":
        phones.pop()
        scales.pop()
    return phones, {"original": text, "model_input": model_input, "rules": rules, "words": word_trace,
                    "duration_scale": scales}
