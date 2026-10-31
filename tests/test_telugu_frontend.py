"""Telugu normalisation and G2P: known conversions, error handling and no character loss."""
import unicodedata

import pytest

from text.telugu import G2PError, cardinal, normalize, ordinal, sentence_to_phones, word_to_phones
from text.telugu import phones as P


@pytest.mark.parametrize("word, expected", [
    ("సమావేశం", "s a m aa v ee sh a m"),            # word-final anusvara -> m
    ("నవంబర్", "n a v a m b a r"),                 # anusvara before b -> m; final virama
    ("పట్టు", "p a tt tt u"),                        # geminate retroflex
    ("విశ్లేషించారు", "v i sh l ee ss i ny c aa r u"),  # anusvara before c -> ny
    ("నిశ్శబ్దంగా", "n i sh sh a b d a ng g aa"),     # anusvara before g -> ng
    ("సిద్ధంగా", "s i d dh a ng g aa"),
    ("ఔషధాల", "au ss a dh aa l a"),                  # independent vowel
    ("ప్రభావాన్ని", "p r a bh aa v aa n n i"),
    ("ఉదయం", "u d a y a m"),
    ("తేదీ", "t ee d ii"),
])
def test_known_words(word, expected):
    assert " ".join(word_to_phones(word)) == expected


@pytest.mark.parametrize("bad", [
    "ింది",       # vowel sign with no consonant (the docx corruption "వచ్చ ింది")
    "్క",          # virama first
    "ంక",          # anusvara first
    "abc",        # non-Telugu letters are not silently dropped
])
def test_invalid_input_raises(bad):
    with pytest.raises(G2PError):
        word_to_phones(bad)


def test_docx_corrupted_text_is_rejected():
    with pytest.raises(G2PError):
        sentence_to_phones("ఆమె పట్టుచీరను ధరించి వేడుకకు వచ్చ ింది.")


@pytest.mark.parametrize("n, words", [
    (0, "సున్నా"), (12, "పన్నెండు"), (45, "నలభై ఐదు"), (100, "వంద"), (120, "వంద ఇరవై"),
    (300, "మూడు వందలు"), (305, "మూడు వందల ఐదు"), (1000, "వెయ్యి"), (2026, "రెండు వేల ఇరవై ఆరు"),
])
def test_cardinals(n, words):
    assert cardinal(n) == words


def test_ordinal_and_time():
    assert ordinal(12) == "పన్నెండవ"
    assert ordinal(2) == "రెండవ"
    assert normalize("12వ తేదీ ఉదయం 10:45 గంటలకు") == "పన్నెండవ తేదీ ఉదయం పది నలభై ఐదు గంటలకు"
    assert normalize("౨౦౨౬") == "రెండు వేల ఇరవై ఆరు"


def test_whitespace_runs_collapse_but_word_boundary_kept():
    assert normalize("అడ  విలో") == "అడ విలో"


def _telugu_letters(text):
    return [c for c in unicodedata.normalize("NFC", text) if "ఀ" <= c <= "౿"]


@pytest.mark.parametrize("sid", ["T1", "T2a", "T2b", "T3", "T4"])
def test_every_test_sentence_converts_without_character_loss(test_items, sid):
    """Each Telugu letter of the (normalised) test sentence is consumed by a G2P rule, and
    every emitted phone belongs to the fixed inventory."""
    phones, trace = sentence_to_phones(test_items[sid]["canonical"])
    inventory = set(P.phone_inventory()) | {"sp"}
    assert phones and set(phones) <= inventory
    model_text = normalize(test_items[sid]["canonical"])
    words = [tok for tok, _ in trace if tok not in {".", ","}]
    assert _telugu_letters("".join(words)) == _telugu_letters(model_text)
    for tok, word_phones in trace:
        if tok not in {".", ","}:
            assert word_phones == word_to_phones(tok)
