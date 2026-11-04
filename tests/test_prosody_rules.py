"""Prosody rules (fastspeech2/prosody_rules.py): pure numpy, no model."""
import pytest

np = pytest.importorskip("numpy")
import prosody_rules as R  # noqa: E402

PH = ["DH", "AH0", "R", "IH1", "DH", "AH0", "M", "AH2", "N", "IH0", "K", "S", "P", "EH1", "K", "T", "L", "IY0"]
WORDS = [{"token": "The", "phones": ["DH", "AH0"]}, {"token": "rhythm", "phones": ["R", "IH1", "DH", "AH0", "M"]},
         {"token": "unexpectedly", "phones": ["AH2", "N", "IH0", "K", "S", "P", "EH1", "K", "T", "L", "IY0"]},
         {"token": ".", "phones": ["sp"]}]


def _pred(final_hz):
    pitch = np.full(len(PH), 210.0)
    pitch[3], pitch[5] = 230.0, 200.0         # rhythm: stressed IH1 above AH0 -> stress rule silent here
    pitch[17] = final_hz                       # final IY0 of the statement
    energy = np.full(len(PH), 50.0)
    energy[3] = 80.0
    dur = np.full(len(PH), 5)
    dur[3] = 9
    return pitch, energy, dur


def test_terminal_fall_lowers_only_a_rising_end():
    p, e, d = _pred(final_hz=243.0)
    pf, ef, df, log = R.factors(PH, WORDS, "The rhythm unexpectedly.", p, e, d, "en", fall=True, stress=0)
    med = np.median(p[[j for j in range(len(PH)) if R.is_vowel(PH[j], "en")]])
    assert pf[17] * p[17] == pytest.approx(R.FALL_RATIO * med)
    assert [x["rule"] for x in log] == ["terminal_fall"] and np.all(pf[:17] == 1)
    p, e, d = _pred(final_hz=160.0)           # already falling: untouched
    pf, _, _, log = R.factors(PH, WORDS, "The rhythm unexpectedly.", p, e, d, "en", fall=True, stress=0)
    assert not log and np.all(pf == 1)


def test_no_fall_on_questions():
    p, e, d = _pred(final_hz=243.0)
    pf, _, _, log = R.factors(PH, WORDS, "The rhythm unexpectedly?", p, e, d, "en", fall=True, stress=0)
    assert not log and np.all(pf == 1)


def test_stress_rule_raises_a_weak_stressed_vowel_with_caps_and_strength():
    p, e, d = _pred(final_hz=160.0)
    p[13], e[13], d[13] = 180.0, 20.0, 3      # EH1 of unexpectedly weaker than IH0 / IY0
    p[9], e[9], d[9] = 220.0, 90.0, 10
    pf, ef, df, log = R.factors(PH, WORDS, "x.", p, e, d, "en", fall=False, stress=1.0)
    assert pf[13] == pytest.approx(R.PITCH_CAP) and ef[13] == pytest.approx(R.ENERGY_CAP)
    assert df[13] == pytest.approx(R.DUR_CAP) and R.DUR_FLOOR <= df[9] < 1
    assert pf[3] == 1 and ef[3] == 1           # rhythm already realised its stress
    half = R.factors(PH, WORDS, "x.", p, e, d, "en", fall=False, stress=0.5)
    assert half[0][13] == pytest.approx(1 + 0.5 * (R.PITCH_CAP - 1))
    off = R.factors(PH, WORDS, "x.", p, e, d, "en", fall=False, stress=0)
    assert all(np.all(a == 1) for a in off[:3]) and not off[3]


def test_telugu_has_no_stress_rule():
    ph = ["p", "a", "t", "t", "u"]
    words = [{"token": "w", "phones": ph}]
    pf, ef, df, log = R.factors(ph, words, "w.", [200, 200, 200, 200, 240], [50] * 5, [5] * 5, "te", stress=1.0)
    assert [x["rule"] for x in log] == ["terminal_fall"] and pf[4] < 1 and np.all(ef == 1) and np.all(df == 1)


def test_stress_realised_counts_and_chance():
    ph = ["K", "AA1", "N", "D", "AH0", "K", "T", "sp", "DH", "AH0"]
    words = [{"token": "conduct", "phones": ph[:7]}, {"token": ",", "phones": ["sp"]}, {"token": "the", "phones": ["DH", "AH0"]}]
    good = R.stress_realised(ph, words, [0, 3, 0, 0, 1, 0, 0, 0, 0, 1], [0, 9, 0, 0, 2, 0, 0, 0, 0, 1], [1, 9, 1, 1, 4, 1, 1, 1, 1, 1])
    bad = R.stress_realised(ph, words, [0, 1, 0, 0, 3, 0, 0, 0, 0, 1], [0, 2, 0, 0, 9, 0, 0, 0, 0, 1], [1, 9, 1, 1, 4, 1, 1, 1, 1, 1])
    assert good[:2] == (1, 1) and bad[:2] == (1, 0)      # "the" has no stressed vowel: not counted
    assert bad[2]["duration"] == 1 and good[2]["chance"] == pytest.approx(0.5)


def _sentence(words_phones):
    words, phones = [], []
    for tok, ph in words_phones:
        words.append({"token": tok, "phones": ph})
        phones += ph
    return phones, words


def test_phrasing_boundaries_exclamation_and_question_rise():
    phones, words = _sentence([("Wait", ["W", "EY1", "T"]), ("!", ["sp"]), ("Are", ["AA1", "R"]),
                               ("you", ["Y", "UW1"]), ("sure", ["SH", "UH1", "R"]), ("?", ["sp"])])
    pitch = np.full(len(phones), 200.0)
    energy = np.full(len(phones), 50.0)
    dur = np.full(len(phones), 6.0)
    dur[3] = 5                                                   # the "!" pause: 5 frames predicted
    pf, ef, df, log = R.factors(phones, words, "Wait! Are you sure?", pitch, energy, dur, "en",
                                fall=True, stress=0, phrasing=True)
    assert dur[3] * df[3] == pytest.approx(R.PAUSE_MIN_FRAMES["!"])        # pause raised to its minimum
    assert df[1] == df[2] == pytest.approx(R.PRE_BOUNDARY) and df[0] == 1  # rhyme of "Wait" lengthened
    assert pf[1] == pytest.approx(R.EMPHASIS_PITCH) and ef[1] == pytest.approx(R.EMPHASIS_ENERGY)
    assert pf[9] * 200 == pytest.approx(R.RISE_RATIO * 200)               # yes/no question: final rise
    assert {x["rule"] for x in log} == {"boundary", "exclamation_emphasis", "question_rise"}


def test_wh_question_falls_and_micro_pause_is_exact():
    phones, words = _sentence([("Why", ["W", "AY1"]), ("<pause>", ["sp"]), ("Monday", ["M", "AH1", "N", "D", "EY2"])])
    pitch = np.array([200, 200, 0, 200, 200, 200, 200, 260.0])
    dur = np.array([5, 9, 3, 4, 5, 4, 4, 9.0])
    pf, _, df, log = R.factors(phones, words, "Why Monday?", pitch, np.full(8, 50.0), dur, "en",
                               fall=True, stress=0, phrasing=True)
    assert dur[2] * df[2] == pytest.approx(R.MICRO_PAUSE_FRAMES)
    assert pf[7] < 1                                                      # wh-question: statement fall
    pf, _, _, _ = R.factors(phones, words, "Why Monday?", pitch, np.full(8, 50.0), dur, "en", fall=True, stress=0)
    assert pf[7] == 1                                                     # without phrasing: as before
