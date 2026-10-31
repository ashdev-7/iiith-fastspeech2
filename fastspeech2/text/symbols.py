""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

from text import cmudict, pinyin
from text.telugu.phones import phone_inventory as _telugu_phone_inventory

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in cmudict.valid_symbols]
_pinyin = ["@" + s for s in pinyin.valid_symbols]

# Export all symbols:
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _arpabet
    + _pinyin
    + _silences
)

# Telugu: rule-based G2P phones (text/telugu/phones.py). A separate table so that its ids,
# and the embedding size of the Telugu model, do not depend on the English/pinyin symbols.
_telugu_symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + ["@" + s for s in _telugu_phone_inventory()]
    + _silences
)

_SYMBOLS_BY_LANGUAGE = {"en": symbols, "te": _telugu_symbols}


def get_symbols(language):
    """Symbol table of a language; index 0 is the padding symbol."""
    if language not in _SYMBOLS_BY_LANGUAGE:
        raise ValueError(f"no symbol table for language {language!r}")
    return _SYMBOLS_BY_LANGUAGE[language]
