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


def test_careful_variant():
    """The citation form is preferred only for these reductions; any other variant keeps the first."""
    from text.english_frontend import careful_variant
    assert careful_variant([["TH", "AW1", "Z", "AH0", "N"], ["TH", "AW1", "Z", "AH0", "N", "D"]])[0][-1] == "D"
    assert careful_variant([["T", "W", "EH1", "N", "IY0"], ["T", "W", "EH1", "N", "T", "IY0"]])[1] == "restored dropped T"
    assert careful_variant([["TH", "ER1", "D", "IY0"], ["TH", "ER1", "T", "IY2"]])[0] == ["TH", "ER1", "T", "IY2"]
    assert careful_variant([["G", "OW1", "IH0", "N"], ["G", "OW1", "IH0", "NG"]])[0][-1] == "NG"
    assert careful_variant([["HH", "W", "AH1", "N"], ["W", "AH1", "N"]])[0] == ["W", "AH1", "N"]
    conduct = [["K", "AA1", "N", "D", "AH0", "K", "T"], ["K", "AH0", "N", "D", "AH1", "K", "T"]]
    assert careful_variant(conduct) == (conduct[0], None)          # stress variants: untouched
    assert careful_variant([["F", "AO1", "R"], ["F", "ER0"]]) == (["F", "AO1", "R"], None)


def test_compound_split():
    from text.english_frontend import compound_split
    lex = {"anti": 1, "disestablishment": 1, "arianism": 1, "dis": 1, "establishment": 1}
    assert compound_split("antidisestablishmentarianism", lex) == ["anti", "disestablishment", "arianism"]
    assert compound_split("establishment", lex) is None            # in the lexicon
    assert compound_split("antidisestablishmentxyz", lex) is None   # a piece is missing
    assert compound_split("antianti", lex) is None                  # shorter than 12 letters


def test_the_before_vowel_careful_forms_and_compounds():
    phones, the = _phones_of("The orchestra played the music.", "The")
    assert the == [["DH", "IY0"]] and _phones_of("The orchestra played the music.", "the")[1] == [["DH", "AH0"]]
    assert phones[:3] == ["DH", "IY0", "AO1"]
    _, w = _phones_of("At eight thirty, one thousand people came.", "thousand")
    assert w == [["TH", "AW1", "Z", "AH0", "N", "D"]]
    assert _phones_of("At eight thirty.", "thirty")[1] == [["TH", "ER1", "T", "IY2"]]
    _, big = _phones_of("The antidisestablishmentarianism movement.", "antidisestablishmentarianism")
    assert big[0][:4] == ["AE1", "N", "T", "AY0"] and big[0][-4:] == ["IH2", "Z", "AH0", "M"]
