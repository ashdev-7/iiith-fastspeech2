"""The canonical test manifest must contain exactly the ten assignment sentences, unaltered."""
import hashlib
import unicodedata

EXPECTED_IDS = ["E1", "E2a", "E2b", "E3", "E4", "T1", "T2a", "T2b", "T3", "T4"]
TELUGU_COMBINING = {chr(c) for c in range(0x0C3E, 0x0C57)} | {"ఁ", "ం", "ః"}


def test_ids_and_languages(test_items):
    assert list(test_items) == EXPECTED_IDS
    assert all(test_items[i]["lang"] == ("en" if i.startswith("E") else "te") for i in EXPECTED_IDS)


def test_hash_nfc_and_codepoints(test_items):
    for item in test_items.values():
        text = item["canonical"]
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == item["canonical_sha256"]
        assert unicodedata.is_normalized("NFC", text)
        assert " ".join(f"{ord(c):04X}" for c in text) == item["canonical_codepoints"]
        assert text == text.strip() and "\n" not in text


def test_no_orphan_telugu_marks(test_items):
    """A dependent vowel sign, virama or anusvara must never follow a space (docx corruption)."""
    for sid in ["T1", "T2a", "T2b", "T3", "T4"]:
        text = test_items[sid]["canonical"]
        for i, ch in enumerate(text):
            if ch in TELUGU_COMBINING:
                assert i > 0 and not text[i - 1].isspace(), f"{sid}: orphan mark at {i}"
        assert "్ం" not in text, f"{sid}: virama followed by anusvara"


def test_deliberate_anomalies_preserved(test_items):
    assert "rhyth  m" in test_items["E3"]["canonical"]
    assert "అడ  విలో" in test_items["T3"]["canonical"]
    assert "andaroo" in test_items["E4"]["canonical"]
