"""Metric definitions used by evaluate_val.py (numpy only)."""
import pytest

np = pytest.importorskip("numpy")

import eval_metrics as M  # noqa: E402


def test_normalise_text_keeps_telugu_marks_and_drops_punctuation():
    assert M.normalise_text("Hello,  World!") == "hello world"
    assert M.normalise_text("పులి అడవిలో నడిచింది.") == "పులి అడవిలో నడిచింది"


def test_error_rates():
    r = M.error_rates("the cat sat", "the cat sat")
    assert r["cer"] == 0 and r["wer"] == 0
    r = M.error_rates("the cat sat", "the bat sat")
    assert r["word_edits"] == 1 and r["char_edits"] == 1 and r["ref_chars"] == 9
    assert abs(r["wer"] - 1 / 3) < 1e-9
    assert M.error_rates("abc", "")["cer"] == 1.0
    rows = [M.error_rates("ab cd", "ab cd"), M.error_rates("ab", "xy")]
    assert abs(M.corpus_rates(rows)["cer"] - 2 / 6) < 1e-9  # total edits / total chars


def test_error_rates_with_extra_normaliser():
    digits = lambda s: s.replace("ten", "10")  # noqa: E731  (stand-in for Whisper's normaliser)
    assert M.error_rates("at ten", "at 10")["cer"] > 0
    assert M.error_rates("at ten", "at 10", norm=digits)["cer"] == 0


def test_levenshtein():
    assert M.levenshtein("kitten", "sitting") == 3
    assert M.levenshtein([], [1, 2]) == 2


def test_mel_cepstra_and_mcd():
    rng = np.random.default_rng(0)
    mel = rng.normal(size=(50, 80))
    c = M.mel_cepstra(mel)
    assert c.shape == (50, 24)
    assert M.mcd_aligned(c, c) == 0.0
    # Orthonormal DCT: a constant offset over all mel bands changes only c0, which MCD excludes.
    assert M.mcd_aligned(M.mel_cepstra(mel + 3.0), c) < 1e-9
    assert M.mcd_aligned(M.mel_cepstra(mel * 1.1), c) > 0


def test_mcd_dtw_with_identity_alignment_matches_aligned():
    c = M.mel_cepstra(np.random.default_rng(1).normal(size=(20, 80)))
    d = c + 0.1
    ident = lambda x, y: [(i, i) for i in range(len(x))]  # noqa: E731
    assert abs(M.mcd_dtw(c, d, dtw=ident)[0] - M.mcd_aligned(c, d)) < 1e-12


def test_pearson_rmse_summarise():
    assert abs(M.pearson([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-12
    assert M.pearson([1, 1, 1], [1, 2, 3]) is None
    assert M.rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))
    s = M.summarise([1.0, None, 3.0])
    assert s["n"] == 2 and s["mean"] == 2.0
    assert M.summarise([None]) == {"n": 0}
