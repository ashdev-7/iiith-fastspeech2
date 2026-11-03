"""Global-variance (GV) post-filter for over-smoothed mel-spectrograms.

FastSpeech 2's decoder is trained with an L1/L2 loss, so on varied data it predicts conditional averages:
the predicted log-mel varies less over time than real speech (measured: English free-running output has
73-89 % of the natural per-bin variance, Telugu 93-98 %; copy-synthesis 95-101 %). The classic remedy from
statistical parametric synthesis (Toda & Tokuda 2007, "global variance") rescales each mel bin's deviation
from its utterance mean so that its variance matches that of natural speech:

    mel'[t, k] = mean_k + (mel[t, k] - mean_k) * r_k ** (strength / 2),   r_k = GV_natural[k] / GV_pred[k]

r_k is estimated once per language on TRAINING utterances of the test voice (ground-truth mel vs the
model's free-running mel of the same text; GV is a per-utterance statistic, so the different lengths do
not matter). strength 0 = off, 1 = full variance match. Inference only; the model is unchanged.
"""
import json

import numpy as np

MAX_SCALE = 1.6  # never amplify a bin's deviations by more than this


def utterance_gv(mel):
    """mel: (T, n_mels) log-mel -> per-bin variance over time."""
    return np.var(np.asarray(mel, dtype=np.float64), axis=0)


def estimate_ratio(gt_mels, pred_mels):
    gt = np.mean([utterance_gv(m) for m in gt_mels], axis=0)
    pr = np.mean([utterance_gv(m) for m in pred_mels], axis=0)
    return (gt / np.maximum(pr, 1e-8)).tolist()


def apply(mel, ratio, strength=1.0):
    """mel: (T, n_mels) numpy or torch tensor (same type returned)."""
    if strength == 0 or ratio is None:
        return mel
    r = np.asarray(ratio, dtype=np.float64)
    scale = np.clip(np.maximum(r, 1.0) ** (0.5 * strength), 1.0, MAX_SCALE)
    try:
        import torch
        if isinstance(mel, torch.Tensor):
            s = torch.as_tensor(scale, dtype=mel.dtype, device=mel.device)
            mu = mel.mean(dim=0, keepdim=True)
            return mu + (mel - mu) * s
    except ImportError:  # pragma: no cover
        pass
    mel = np.asarray(mel)
    mu = mel.mean(axis=0, keepdims=True)
    return (mu + (mel - mu) * scale).astype(mel.dtype)


def load_ratio(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["ratio"]
