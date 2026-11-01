"""English words written in Latin script inside Telugu text (code-mixing, e.g. "నేను iPhone కొన్నాను").

Before this, such words raised G2PError. Now: English pronunciation (the English frontend's lexicon
lookup, else g2p_en) -> ARPAbet without stress -> the nearest phones of the Telugu inventory,
following how Telugu speakers write and say English loanwords (alveolar T/D -> retroflex tt/dd as in
"టీవీ"; F -> ph as in "ఫోన్"; Z -> j; W -> v; /ae/ after a consonant -> "y aa" as in "బ్యాంక్" for "bank").

Only phones that already exist in the Telugu symbol table (phones.phone_inventory) are produced, so
the trained model and its vocabulary are unchanged. Training transcripts contain no Latin script, so
the training data and the assignment's test sentences are unaffected. This is an approximation for
robustness, not a verified phonetic analysis.
"""
import os
import re

ARPABET_TO_TELUGU = {
    "AA": ["aa"], "AE": ["aa"], "AH": ["a"], "AO": ["aa"], "AW": ["au"], "AY": ["ai"], "EH": ["e"],
    "ER": ["a", "r"], "EY": ["ee"], "IH": ["i"], "IY": ["ii"], "OW": ["oo"], "OY": ["aa", "y"],
    "UH": ["u"], "UW": ["uu"],
    "B": ["b"], "CH": ["c"], "D": ["dd"], "DH": ["d"], "F": ["ph"], "G": ["g"], "HH": ["h"], "JH": ["j"],
    "K": ["k"], "L": ["l"], "M": ["m"], "N": ["n"], "NG": ["ng"], "P": ["p"], "R": ["r"], "S": ["s"],
    "SH": ["sh"], "T": ["tt"], "TH": ["th"], "V": ["v"], "W": ["v"], "Y": ["y"], "Z": ["j"], "ZH": ["j"],
}
_VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
LEXICON = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "lexicon", "librispeech-lexicon.txt")


def arpabet_to_telugu(arpabet):
    """['AY1', 'F', 'OW0', 'N'] -> ['ai', 'ph', 'oo', 'n']"""
    out, prev = [], None
    for p in arpabet:
        base = p.rstrip("012")
        if base not in ARPABET_TO_TELUGU:
            continue  # "sp" or punctuation from the English frontend
        if base == "AE" and prev is not None and prev not in _VOWELS:
            out.extend(["y", "aa"])  # bank -> b y aa ng k
        else:
            out.extend(ARPABET_TO_TELUGU[base])
        prev = base
    return out


_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+")


def camel_parts(word):
    """'iPhone' -> ['i', 'Phone'], 'eBay' -> ['e', 'Bay'], 'YouTube' -> ['You', 'Tube'], 'laptop' -> ['laptop'].
    g2p_en reads unseen brand names letter-blind ('iPhone' -> IH F OW N); a lone letter part is
    looked up as a word, which gives its letter name (I -> AY1)."""
    return _CAMEL_RE.findall(word) or [word]


def english_arpabet(word):
    from text.english_frontend import text_to_phones  # lazy: needs g2p_en only when Latin appears
    out = []
    for part in camel_parts(word):
        phones, _ = text_to_phones(part, LEXICON)
        out.extend(p for p in phones if p != "sp")
    return out


def latin_word_to_phones(word):
    return arpabet_to_telugu(english_arpabet(word))
