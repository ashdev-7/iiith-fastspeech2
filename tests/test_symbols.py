"""Per-language symbol tables and token-id conversion."""
import pytest

from text import sequence_to_text, text_to_sequence
from text.symbols import get_symbols, symbols
from text.telugu import phone_inventory, sentence_to_phones


def test_english_table_is_the_reference_table():
    assert get_symbols("en") is symbols


def test_telugu_table_covers_inventory_and_silences():
    table = get_symbols("te")
    assert table[0] == "_" and len(table) == len(set(table))
    assert {"@" + p for p in phone_inventory()} | {"@sp", "@spn", "@sil"} <= set(table)


def test_telugu_round_trip(test_items):
    phones, _ = sentence_to_phones(test_items["T4"]["canonical"])
    ids = text_to_sequence("{" + " ".join(phones) + "}", [], "te")
    assert len(ids) == len(phones) and 0 not in ids
    assert sequence_to_text(ids, "te") == "{" + " ".join(phones) + "}"


def test_unknown_symbol_fails_loudly():
    with pytest.raises(ValueError):
        text_to_sequence("{AA1 K}", [], "te")  # ARPAbet phones are not Telugu symbols
