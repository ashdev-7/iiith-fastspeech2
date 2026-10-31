"""Telugu grapheme -> phone tables used by the rule-based G2P (text/telugu/g2p.py).

Phone labels are lowercase ASCII so that they can be written into MFA pronunciation
dictionaries and TextGrids unchanged. Conventions:
  * doubled letters mark retroflex consonants (tt, dd, nn, ss, ll) and long vowels (aa, ii, ...);
  * a trailing "h" marks aspiration (kh, gh, ch, ...);
  * "c" is the palatal stop (చ), "sh" the palatal sibilant (శ), "ss" the retroflex sibilant (ష).

The labels are a transcription of the script; they are not a verified phonetic analysis
of the IndicTTS speakers' speech (e.g. ఫ in English loanwords is often realised as /f/
but is transcribed "ph" here, as written).
"""

# Independent vowels (U+0C05..U+0C14, U+0C60, U+0C61).
INDEPENDENT_VOWELS = {
    "అ": ["a"], "ఆ": ["aa"], "ఇ": ["i"], "ఈ": ["ii"],
    "ఉ": ["u"], "ఊ": ["uu"], "ఋ": ["r", "u"], "ౠ": ["r", "uu"],
    "ఌ": ["l", "u"], "ౡ": ["l", "uu"],
    "ఎ": ["e"], "ఏ": ["ee"], "ఐ": ["ai"],
    "ఒ": ["o"], "ఓ": ["oo"], "ఔ": ["au"],
}

# Dependent vowel signs (matras); they replace the inherent vowel of the preceding consonant.
VOWEL_SIGNS = {
    "ా": ["aa"], "ి": ["i"], "ీ": ["ii"], "ు": ["u"], "ూ": ["uu"],
    "ృ": ["r", "u"], "ౄ": ["r", "uu"], "ౢ": ["l", "u"], "ౣ": ["l", "uu"],
    "ె": ["e"], "ే": ["ee"], "ై": ["ai"],
    "ొ": ["o"], "ో": ["oo"], "ౌ": ["au"],
}

# Consonants (U+0C15..U+0C39 and the historic letters U+0C58..U+0C5A).
CONSONANTS = {
    "క": "k", "ఖ": "kh", "గ": "g", "ఘ": "gh", "ఙ": "ng",
    "చ": "c", "ఛ": "ch", "జ": "j", "ఝ": "jh", "ఞ": "ny",
    "ట": "tt", "ఠ": "tth", "డ": "dd", "ఢ": "ddh", "ణ": "nn",
    "త": "t", "థ": "th", "ద": "d", "ధ": "dh", "న": "n",
    "ప": "p", "ఫ": "ph", "బ": "b", "భ": "bh", "మ": "m",
    "య": "y", "ర": "r", "ఱ": "rr", "ల": "l", "ళ": "ll", "ఴ": "zh",
    "వ": "v", "శ": "sh", "ష": "ss", "స": "s", "హ": "h",
    "ౘ": "ts", "ౙ": "dz", "ౚ": "rr",
}

VIRAMA = "్"
ANUSVARA = "ం"
VISARGA = "ః"
CANDRABINDU = "ఁ"  # arasunna: archaic nasalisation mark, not pronounced as a separate phone
NUKTA = "఼"
AVAGRAHA = "ఽ"
LENGTH_MARKS = {"ౕ", "ౖ"}  # only occur inside decomposed long vowels; NFC folds them
INHERENT_VOWEL = "a"

# Homorganic nasal chosen for an anusvara by the place of articulation of the next consonant.
_PLACE_NASAL = {
    "ng": ("k", "kh", "g", "gh", "ng"),
    "ny": ("c", "ch", "j", "jh", "ny"),
    "nn": ("tt", "tth", "dd", "ddh", "nn"),
    "n": ("t", "th", "d", "dh", "n"),
    "m": ("p", "ph", "b", "bh", "m"),
}
ANUSVARA_BEFORE = {c: nasal for nasal, group in _PLACE_NASAL.items() for c in group}
ANUSVARA_DEFAULT = "m"  # word-final or before a non-stop consonant (e.g. సమావేశం -> ...am)

SILENCES = ["sp", "spn", "sil"]


def phone_inventory():
    """Every phone label the G2P can emit, in a fixed order (used for the symbol table)."""
    phones = []
    for seq in list(INDEPENDENT_VOWELS.values()) + list(VOWEL_SIGNS.values()):
        phones.extend(seq)
    phones.extend(CONSONANTS.values())
    phones.extend(ANUSVARA_BEFORE.values())
    phones.extend([INHERENT_VOWEL, ANUSVARA_DEFAULT, "h"])
    return sorted(set(phones))
