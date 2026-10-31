"""English normalisation (model input only); pronunciation checks need g2p_en (run on Kaggle)."""
import os

import pytest

from text.english_frontend import normalize


def test_e4_time_date_year():
    out, rules = normalize("The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.")
    assert out == ("The meeting will start at ten forty five A M on twelve November twenty twenty six, "
                   "so andaroo should arrive on time.")
    assert rules == ["time", "numbers", "hyphen"]


def test_split_token_is_not_merged():
    assert normalize("The rhyth  m of the orchestra")[0] == "The rhyth m of the orchestra"


@pytest.mark.parametrize("src,expected", [("9:05 pm", "nine oh five P M"), ("12:00", "twelve o'clock")])
def test_times(src, expected):
    assert normalize(src)[0] == expected


@pytest.mark.observation
def test_pronunciations(test_items):
    """Observation, not a gate: g2p_en's POS tagger may mis-tag; the result is reported by job 03."""
    pytest.importorskip("g2p_en")
    from text.english_frontend import text_to_phones
    LEXICON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "fastspeech2", "lexicon", "librispeech-lexicon.txt")
    items = {k: v["canonical"] for k, v in test_items.items()}
    _, a = text_to_phones(items["E2a"], LEXICON)
    _, b = text_to_phones(items["E2b"], LEXICON)
    conduct = lambda tr: next(w for w in tr["words"] if w["token"].lower() == "conduct")  # noqa: E731
    # noun: primary stress on the 1st syllable; verb: on the 2nd (vowels carry the stress digit)
    stress = lambda tr: [p[-1] for p in conduct(tr)["phones"] if p[-1] in "012"]  # noqa: E731
    assert stress(a) == ["1", "0"], conduct(a)
    assert stress(b) == ["0", "1"], conduct(b)
    phones, e3 = text_to_phones(items["E3"], LEXICON)
    assert [w["token"] for w in e3["words"]][:3] == ["The", "rhyth", "m"]
    assert all(p != "sp" for p in phones[:6])  # no pause invented inside the sentence start
