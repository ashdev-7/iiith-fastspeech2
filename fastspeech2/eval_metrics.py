"""Objective metrics for evaluate_val.py (numpy only, so they are unit-tested without torch).

Definitions (stated in the report exactly as implemented here):
  * CER / WER: Levenshtein distance / reference length, after normalise_text() on both sides.
  * mel-cepstral distortion (MCD-mel): the 80-band natural-log mel spectrogram (the model's own
    feature) is turned into cepstra with an orthonormal DCT-II; coefficients c1..c24 (c0 = energy is
    excluded) give  MCD = (10 / ln 10) * sqrt(2 * sum_k (c_k - c'_k)^2), averaged over frames.
    This is NOT the WORLD/SPTK mel-cepstrum MCD that papers report; values are only comparable
    within this project (between models, languages and conditions computed the same way).
  * Pearson r and RMSE on phone-level values.
"""
import re
import unicodedata

import numpy as np

MCD_K = 10.0 / np.log(10.0) * np.sqrt(2.0)


def normalise_text(text):
    """NFC, lower case, punctuation/symbols -> space, collapsed whitespace (Telugu marks are kept:
    vowel signs and virama are category M, not P/S)."""
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)
    return re.sub(r"\s+", " ", text).strip()


def levenshtein(a, b):
    """Edit distance between two sequences (insertions, deletions, substitutions cost 1)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def error_rates(reference, hypothesis):
    """{"cer", "wer", "ref_chars", "ref_words", "char_edits", "word_edits"} after normalisation.
    CER counts characters without spaces."""
    r, h = normalise_text(reference), normalise_text(hypothesis)
    rc, hc = r.replace(" ", ""), h.replace(" ", "")
    rw, hw = r.split(), h.split()
    ce, we = levenshtein(rc, hc), levenshtein(rw, hw)
    return {"cer": ce / max(1, len(rc)), "wer": we / max(1, len(rw)), "ref_chars": len(rc), "ref_words": len(rw),
            "char_edits": ce, "word_edits": we}


def corpus_rates(rows):
    """Corpus-level CER/WER = total edits / total reference length (not a mean of ratios)."""
    ce = sum(r["char_edits"] for r in rows)
    we = sum(r["word_edits"] for r in rows)
    return {"cer": ce / max(1, sum(r["ref_chars"] for r in rows)),
            "wer": we / max(1, sum(r["ref_words"] for r in rows)), "utterances": len(rows)}


def mel_cepstra(logmel, n=24):
    """(T, n_mels) natural-log mel -> (T, n) cepstra c1..cn (orthonormal DCT-II over the mel axis)."""
    logmel = np.asarray(logmel, dtype=np.float64)
    m = logmel.shape[1]
    k = np.arange(m)
    basis = np.cos(np.pi / m * (k[None, :] + 0.5) * k[:, None]) * np.sqrt(2.0 / m)
    basis[0] /= np.sqrt(2.0)
    return (logmel @ basis.T)[:, 1:n + 1]


def mcd_aligned(c_a, c_b):
    """Mean MCD over frame-aligned cepstra (same length)."""
    d = np.asarray(c_a) - np.asarray(c_b)
    return float(np.mean(MCD_K * np.sqrt(np.sum(d * d, axis=1))))


def mcd_dtw(c_a, c_b, dtw=None):
    """MCD after DTW alignment (for free-running synthesis whose length differs from the reference).
    dtw: function(X, Y) -> warping path [(i, j), ...]; defaults to librosa.sequence.dtw.
    Returns (mcd, path_length)."""
    if dtw is None:
        import librosa

        def dtw(x, y):
            _, wp = librosa.sequence.dtw(X=x.T, Y=y.T, metric="euclidean")
            return wp[::-1]
    path = np.asarray(dtw(np.asarray(c_a), np.asarray(c_b)))
    return mcd_aligned(np.asarray(c_a)[path[:, 0]], np.asarray(c_b)[path[:, 1]]), int(len(path))


def pearson(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def rmse(x, y):
    d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return float(np.sqrt(np.mean(d * d))) if len(d) else None


def summarise(values):
    """mean / median / std / n of the non-None values."""
    v = np.array([x for x in values if x is not None], dtype=np.float64)
    if not len(v):
        return {"n": 0}
    return {"mean": float(v.mean()), "median": float(np.median(v)), "std": float(v.std()), "n": int(len(v))}
