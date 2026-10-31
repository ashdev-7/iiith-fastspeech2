"""Rule-based Telugu grapheme-to-phone conversion.

Telugu orthography is close to phonemic, so each akshara maps to phones deterministically:
  * a consonant carries the inherent vowel /a/ unless followed by a vowel sign or virama;
  * a vowel sign replaces the inherent vowel; virama removes it;
  * anusvara becomes the nasal homorganic with the next consonant, otherwise /m/;
  * visarga becomes /h/; candrabindu, nukta and avagraha produce no phone.
Every Telugu code point must be consumed by exactly one rule. Anything else (a vowel sign
or virama without a preceding consonant, or an unknown character) raises G2PError, so
characters are never dropped silently.
"""
import re
import unicodedata

from . import phones as P
from .normalize import normalize

PUNCTUATION = set(".,!?;:-\"'()[]{}।॥…–—“”‘’")
_SPLIT_RE = re.compile(r"(\s+|[" + re.escape("".join(sorted(PUNCTUATION))) + r"])")


class G2PError(ValueError):
    pass


def word_to_phones(word):
    """Convert one Telugu word (no spaces or punctuation) to a list of phone labels."""
    word = unicodedata.normalize("NFC", word)
    out = []
    pending = False  # the last emitted consonant still owes a vowel
    for i, ch in enumerate(word):
        if ch in P.CONSONANTS:
            if pending:
                out.append(P.INHERENT_VOWEL)
            out.append(P.CONSONANTS[ch])
            pending = True
        elif ch in P.VOWEL_SIGNS or ch == P.VIRAMA:
            if not pending:
                raise G2PError(f"{'vowel sign' if ch != P.VIRAMA else 'virama'} U+{ord(ch):04X} "
                               f"without a preceding consonant at index {i} in {word!r}")
            out.extend(P.VOWEL_SIGNS.get(ch, []))
            pending = False
        elif ch in P.INDEPENDENT_VOWELS:
            if pending:
                out.append(P.INHERENT_VOWEL)
            out.extend(P.INDEPENDENT_VOWELS[ch])
            pending = False
        elif ch == P.ANUSVARA:
            if pending:
                out.append(P.INHERENT_VOWEL)
                pending = False
            elif not out:
                raise G2PError(f"anusvara at the start of {word!r}")
            nxt = word[i + 1] if i + 1 < len(word) else ""
            out.append(P.ANUSVARA_BEFORE.get(P.CONSONANTS.get(nxt, ""), P.ANUSVARA_DEFAULT))
        elif ch == P.VISARGA:
            if pending:
                out.append(P.INHERENT_VOWEL)
                pending = False
            out.append("h")
        elif ch == P.NUKTA:
            if not pending:
                raise G2PError(f"nukta without a preceding consonant in {word!r}")
        elif ch in (P.CANDRABINDU, P.AVAGRAHA) or ch in P.LENGTH_MARKS:
            if pending and ch == P.CANDRABINDU:
                out.append(P.INHERENT_VOWEL)
                pending = False
        else:
            raise G2PError(f"unsupported character U+{ord(ch):04X} "
                           f"({unicodedata.name(ch, '?')}) in {word!r}")
    if pending:
        out.append(P.INHERENT_VOWEL)
    return out


def sentence_to_phones(text, do_normalize=True):
    """Convert a sentence to phones, with punctuation mapped to the short pause "sp".

    Returns (phones, trace); trace is a list of (token, phones) pairs for auditing.
    Raises G2PError on any character that the rules cannot consume.
    """
    model_text = normalize(text) if do_normalize else text
    phones, trace = [], []
    for token in _SPLIT_RE.split(model_text):
        if not token or token.isspace():
            continue
        if token in PUNCTUATION:
            if phones and phones[-1] != "sp":
                phones.append("sp")
            trace.append((token, ["sp"]))
            continue
        word_phones = word_to_phones(token)
        phones.extend(word_phones)
        trace.append((token, word_phones))
    while phones and phones[-1] == "sp":
        phones.pop()
    return phones, trace


def words_for_lexicon(text, do_normalize=True):
    """Yield (word, phones) for every word of a transcript, for building an MFA dictionary."""
    for token, word_phones in sentence_to_phones(text, do_normalize)[1]:
        if token not in PUNCTUATION:
            yield token, word_phones
