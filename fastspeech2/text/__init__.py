""" from https://github.com/keithito/tacotron """
import re
from text import cleaners
from text.symbols import get_symbols


# Mappings from symbol to numeric ID and vice versa, one pair per language:
_maps = {}


def _symbol_maps(language):
    if language not in _maps:
        table = get_symbols(language)
        _maps[language] = ({s: i for i, s in enumerate(table)}, {i: s for i, s in enumerate(table)})
    return _maps[language]


# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")


def text_to_sequence(text, cleaner_names, language="en"):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

    The text can optionally have ARPAbet sequences enclosed in curly braces embedded
    in it. For example, "Turn left on {HH AW1 S S T AH0 N} Street."

    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
      language: selects the symbol table ("en" or "te")

    Returns:
      List of integers corresponding to the symbols in the text
    """
    symbol_to_id = _symbol_maps(language)[0]
    sequence = []

    # Check for curly braces and treat their contents as ARPAbet:
    while len(text):
        m = _curly_re.match(text)

        if not m:
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names), symbol_to_id)
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names), symbol_to_id)
        sequence += _arpabet_to_sequence(m.group(2), symbol_to_id)
        text = m.group(3)

    return sequence


def sequence_to_text(sequence, language="en"):
    """Converts a sequence of IDs back to a string"""
    id_to_symbol = _symbol_maps(language)[1]
    result = ""
    for symbol_id in sequence:
        if symbol_id in id_to_symbol:
            s = id_to_symbol[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == "@":
                s = "{%s}" % s[1:]
            result += s
    return result.replace("}{", " ")


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text


def _symbols_to_sequence(symbols, symbol_to_id):
    unknown = [s for s in symbols if s not in symbol_to_id and s not in ("_", "~")]
    if unknown:
        # The reference silently dropped unknown symbols; fail loudly instead so that a
        # frontend/symbol-table mismatch cannot pass unnoticed.
        raise ValueError(f"symbols not in the symbol table: {sorted(set(unknown))}")
    return [symbol_to_id[s] for s in symbols if _should_keep_symbol(s, symbol_to_id)]


def _arpabet_to_sequence(text, symbol_to_id):
    return _symbols_to_sequence(["@" + s for s in text.split()], symbol_to_id)


def _should_keep_symbol(s, symbol_to_id):
    return s in symbol_to_id and s != "_" and s != "~"
