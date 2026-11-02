"""English inference frontend: original text -> model input text -> ARPAbet phones (plan.txt 5.3).

The original text is never modified; every step is recorded in the returned trace.
Normalisation (model input only):
  * times "H:MM" (optionally followed by AM/PM) -> "ten forty five AM", AM/PM read as letter names (minutes 01-09 -> "oh five",
    00 -> "o'clock");
  * other numbers: the reference's inflect-based normalize_numbers (years 1000-2999 are read in
    pairs, e.g. 2026 -> "twenty twenty six"; days are cardinal, as written: "12 November" -> "twelve");
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
  * punctuation -> "sp", as in the reference.
"""
import re

from text.numbers import normalize_numbers

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

_lexicon = None
_g2p = None


def _number_words(n):
    return normalize_numbers(str(n))


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


def load_lexicon(path):
    lex = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                lex.setdefault(parts[0].lower(), parts[1:])  # first pronunciation, as in the reference
    return lex


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
    if any(w.lower() in g2p.homograph2features for w in words):
        from nltk import pos_tag
        tags = {i: tag for i, (_, tag) in enumerate(pos_tag(words))}

    phones, scales, word_trace, wi = [], [], [], 0
    for tok in tokens:
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
        elif low in g2p.homograph2features:
            pron1, pron2, pos1 = g2p.homograph2features[low]
            pron = pron1 if tags[wi].startswith(pos1) else pron2
            source = "g2p_en homograph (POS {})".format(tags[wi])
        elif low in _lexicon:
            pron, source = _lexicon[low], "lexicon"
        else:
            pron = [p for p in g2p(tok) if re.match(r"^[A-Z]+[012]?$", p)]
            source = "g2p_en (out of lexicon)"
            pron, fixed = fix_final_oo(tok, pron)
            if fixed:
                source += ", final 'oo' -> UW"
        phones.extend(pron)
        scale = LETTER_DURATION_SCALE if source == "letter names" else 1.0
        scales.extend([scale] * len(pron))
        word_trace.append({"token": tok, "source": source, "phones": list(pron)})
        if scale != 1.0:
            word_trace[-1]["duration_scale"] = scale
        wi += 1
    while phones and phones[-1] == "sp":
        phones.pop()
        scales.pop()
    return phones, {"original": text, "model_input": model_input, "rules": rules, "words": word_trace,
                    "duration_scale": scales}
