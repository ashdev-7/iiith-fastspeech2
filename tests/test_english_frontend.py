"""English normalisation (model input only); pronunciation checks need g2p_en (run on Kaggle)."""
import os

import pytest

from text.english_frontend import normalize


def test_e4_time_date_year():
    out, rules = normalize("The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.")
    assert out == ("The meeting will start at ten forty five AM on twelve November twenty twenty six, "
                   "so andaroo should arrive on time.")
    assert rules == ["time", "numbers", "hyphen"]


def test_normalisation_does_not_merge_split_tokens():
    # The model-input text keeps "rhyth m"; joining happens only for the pronunciation lookup.
    assert normalize("The rhyth  m of the orchestra")[0] == "The rhyth m of the orchestra"


@pytest.mark.parametrize("src,expected", [("9:05 pm", "nine oh five PM"), ("12:00", "twelve o'clock")])
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
    assert [w["token"] for w in e3["words"]][:2] == ["The", "rhythm"]  # split word joined for pronunciation
    assert e3["original"] == items["E3"]                                 # the text itself is unchanged
    assert all(p != "sp" for p in phones[:6])  # no pause invented inside the sentence start
    _, e4 = text_to_phones(items["E4"], LEXICON)
    andaroo = next(w for w in e4["words"] if w["token"] == "andaroo")
    assert andaroo["phones"][-1].startswith("UW"), andaroo


def test_join_split_words():
    from text.english_frontend import join_split_words
    lex = {"the": [], "rhythm": [], "m": [], "of": []}
    toks, rules = join_split_words(["The", "rhyth", "m", "of"], lex)
    assert toks == ["The", "rhythm", "of"] and rules == ["joined split word 'rhyth m' -> 'rhythm'"]
    # both parts real words ("the" + "m" is not a word anyway): never joined
    assert join_split_words(["of", "the"], {"of": [], "the": [], "ofthe": []})[0] == ["of", "the"]
    assert join_split_words(["rhyth", ","], lex)[0] == ["rhyth", ","]  # punctuation never joined


def test_fix_final_oo():
    from text.english_frontend import fix_final_oo
    assert fix_final_oo("andaroo", ["AE0", "N", "D", "AA1", "R", "OW0"]) == (["AE0", "N", "D", "AA1", "R", "UW0"], True)
    assert fix_final_oo("bamboo", ["B", "AE0", "M", "B", "UW1"])[1] is False
    assert fix_final_oo("hello", ["HH", "AH0", "L", "OW1"])[1] is False


def test_am_pm_are_letter_names():
    pytest.importorskip("g2p_en")
    from text.english_frontend import text_to_phones
    lex = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "fastspeech2", "lexicon", "librispeech-lexicon.txt")
    _, tr = text_to_phones("start at 10:45 AM or 9:05 pm", lex)
    by_token = {w["token"]: w["phones"] for w in tr["words"]}
    assert by_token["AM"] == ["EY1", "EH1", "M"] and by_token["PM"] == ["P", "IY1", "EH1", "M"]
    # letter names are slowed down per phone; every other phone keeps scale 1
    phones, tr2 = text_to_phones("The meeting starts at 10:45 AM, so be on time.", lex)
    assert len(tr2["duration_scale"]) == len(phones)
    am = next(w for w in tr2["words"] if w["token"] == "AM")
    assert am["duration_scale"] > 1
    assert sum(1 for x in tr2["duration_scale"] if x != 1.0) == len(am["phones"])


@pytest.mark.parametrize("src,expected", [
    ("at 9 a.m.", "at nine AM"), ("from 7 pm", "from seven PM"), ("10 AM", "ten AM"),
    ("20% off", "twenty percent off"), ("R&D", "R and D"), ("I am here", "I am here"),
])
def test_hour_ampm_and_symbols(src, expected):
    assert normalize(src)[0] == expected


def test_acronyms():
    """Two capitals are spelled (UK); 3-6 use CMUdict if listed (NASA is a word) and are spelled otherwise
    (IIIT, as in stress sentence SE06, was read as "eye-it")."""
    pytest.importorskip("g2p_en")
    from text.english_frontend import text_to_phones
    lex = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "fastspeech2", "lexicon", "librispeech-lexicon.txt")
    _, tr = text_to_phones("NASA and the UK at IIIT Hyderabad.", lex)
    w = {x["token"]: x for x in tr["words"]}
    assert w["NASA"]["phones"] == ["N", "AE1", "S", "AH0"]
    assert w["UK"]["phones"] == ["Y", "UW1", "K", "EY1"] and w["UK"]["duration_scale"] > 1
    assert w["IIIT"]["phones"] == ["AY1", "AY1", "AY1", "T", "IY1"]
    assert w["Hyderabad"]["source"] != "letter names"


@pytest.mark.parametrize("src,expected", [
    ("approximately 1,247 researchers", "approximately one thousand two hundred forty seven researchers"),
    ("born in 1947", "born in nineteen forty seven"),
    ("on November 12, 2026", "on November twelfth, twenty twenty six"),
    ("on 12 November 2026", "on twelve November twenty twenty six"),   # assignment E4 order: unchanged
])
def test_thousands_separators_and_month_day(src, expected):
    assert normalize(src)[0] == expected


def _phones_of(text, token):
    pytest.importorskip("g2p_en")
    from text.english_frontend import text_to_phones
    lex = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "fastspeech2", "lexicon", "librispeech-lexicon.txt")
    phones, tr = text_to_phones(text, lex)
    return phones, [w["phones"] for w in tr["words"] if w["token"] == token]


def test_read_tense_and_process():
    """g2p_en's homograph table maps past-tense 'read' to R IY1 D and gives the verb 'process' the
    procession sense; both are corrected. A past-time word in the clause marks 'read' as past."""
    _, r = _phones_of("I read the book yesterday, and today I will read another chapter.", "read")
    assert r == [["R", "EH1", "D"], ["R", "IY1", "D"]]
    assert _phones_of("I read books every day.", "read")[1] == [["R", "IY1", "D"]]
    assert _phones_of("I have read it.", "read")[1] == [["R", "EH1", "D"]]
    assert _phones_of("We can process millions of records.", "process")[1][0][:4] == ["P", "R", "AA1", "S"]


def test_no_pause_after_title_abbreviation():
    phones, dr = _phones_of("Dr. Williams asked. Then he left.", "Dr")
    assert dr == [["D", "AA1", "K", "T", "ER0"]]
    assert phones[:6] == ["D", "AA1", "K", "T", "ER0", "W"]   # no sp between "Dr." and "Williams"
    assert "sp" in phones                                      # the sentence period still pauses
