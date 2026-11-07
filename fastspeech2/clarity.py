"""Inference-time "clear speech" options (no retraining; the model and its weights are unchanged).

Measured problems and the remedy for each (report §5.3):
  * Endings cut off. Training clips are trimmed at the first/last phone, so the model's last frame is at
    full speech level (mean log-mel about -5.3, digital silence is -11.5) and the vocoder output stops
    mid-sound. edges(): a short lead-in of silence, a 5-frame log-domain fade after the last frame and
    trailing silence, before the vocoder.
  * Background noise. The vocoder is clean (-83 dBFS on a silent mel) but the English model reproduces
    the Svarah recordings' noise floor: in its pauses every mel band sits near -6.5 instead of -11.5.
    denoise(): spectral subtraction of that floor in the mel domain (power subtraction with a -20 dB
    gain floor, gains smoothed over 3 frames). The profile is measured on the model's own pauses in
    neutral sentences that are not test sentences (estimate_noise_profile, NOISE_TEXTS).
  * Fast, run-together speech. On validation the model is 4 % shorter than the real speakers (median
    length ratio 0.960); clear speech is also slower and separates heavy words. prepare(): a global
    tempo factor, extra time for long words (>= LONG_WORD_VOWELS vowels) and a micro-pause before them
    (its length is set in prosody_rules rule 3).
"""
import json

import numpy as np

SIL = float(np.log(1e-5))      # log-mel of digital silence (the mel extractor clamps at 1e-5)
LONG_WORD_VOWELS = 4
NOISE_TEXTS = ["Yes, of course, we can talk about it later.", "First, open the box; then, take out the papers.",
               "Well, I think so, but, honestly, I am not sure."]
SILENCES = {"sp", "spn", "sil"}
FUNCTION_WORDS = set("the a an of to in on at for by with from and or but as this that these those his her its "
                     "their our my your".split())


def _vowels(phones, idx, lang):
    import prosody_rules
    return sum(prosody_rules.is_vowel(phones[j], lang) for j in idx)


def prepare(phones, trace, lang, tempo=1.0, long_word_scale=1.0, heavy_pause=False):
    """Returns (phones, trace, factors). With heavy_pause, a "<pause>" micro-pause is inserted before each long
    word, ahead of its preceding function words ("during <pause> the parliamentary"; the model lengthens the
    vowel before a pause, so "the <pause> parliamentary" would give a drawn-out "the"); none at the sentence
    start or next to an existing pause. factors: per-phone duration factors (duration_scale x tempo x long-word
    scale)."""
    import prosody_rules
    scales = list(trace.get("duration_scale") or [1.0] * len(phones))
    words = trace["words"]
    if heavy_pause:
        spans = prosody_rules.align_words(phones, words)
        heavy = set()
        for k, (_, idx) in enumerate(spans):
            if _vowels(phones, idx, lang) < LONG_WORD_VOWELS:
                continue
            while k > 0 and spans[k - 1][0].lower() in FUNCTION_WORDS:  # pause before "the parliamentary",
                k -= 1                                                    # never between "the" and its noun
            j = spans[k][1][0]
            if j > 0 and phones[j - 1] not in SILENCES:
                heavy.add(j)
        if heavy:
            ph, sc = [], []
            for j, p in enumerate(phones):
                if j in heavy:
                    ph.append("sp")
                    sc.append(1.0)
                ph.append(p)
                sc.append(scales[j])
            new_words, k = [], 0  # k: index into spans (one per word with non-silence phones, in order)
            for w in words:
                if w["phones"] and not all(p in SILENCES for p in w["phones"]):
                    if spans[k][1][0] in heavy:
                        new_words.append({"token": "<pause>", "source": "clarity micro-pause", "phones": ["sp"]})
                    k += 1
                new_words.append(w)
            phones, scales, trace = ph, sc, dict(trace, words=new_words)
    f = np.asarray(scales, dtype=float) * tempo
    if long_word_scale != 1.0:
        for _, idx in prosody_rules.align_words(phones, trace["words"]):
            if _vowels(phones, idx, lang) >= LONG_WORD_VOWELS:
                f[idx] *= long_word_scale
    trace = dict(trace, duration_scale=scales, clarity={"tempo": tempo, "long_word_scale": long_word_scale,
                                                        "heavy_word_pause": heavy_pause})
    return phones, trace, f


def denoise(mel, profile, alpha=1.0, floor=0.1):
    """mel (T, n_mels) torch log-mel; profile (n_mels,) log-mel noise floor."""
    import torch
    prof = torch.as_tensor(np.asarray(profile), dtype=mel.dtype, device=mel.device)
    g = torch.clamp(1 - alpha * torch.exp(2 * (prof - mel)), min=floor ** 2).sqrt()
    g = torch.nn.functional.avg_pool1d(g.T.unsqueeze(0), 3, 1, 1, count_include_pad=False)[0].T
    return mel + torch.log(g)


def edges(mel, lead=4, fade=5, tail=10):
    """Silence lead-in, log-domain fade of the last frame to silence, trailing silence. Returns (mel, lead)."""
    import torch
    n = mel.shape[1]
    w = torch.linspace(1, 0, fade + 2, dtype=mel.dtype, device=mel.device)[1:-1].unsqueeze(1)
    sil = lambda k: torch.full((k, n), SIL, dtype=mel.dtype, device=mel.device)  # noqa: E731
    return torch.cat([sil(lead), mel, mel[-1:] * w + SIL * (1 - w), sil(tail)]), lead


def load_profile(path):
    with open(path, encoding="utf-8") as f:
        return np.asarray(json.load(f)["profile"], dtype=np.float32)


def estimate_noise_profile(synth_mel_and_durations, frontend, lang):
    """Median log-mel of the model's pause frames (>= 6 frames, 2 trimmed at each side) over NOISE_TEXTS.
    synth_mel_and_durations(phones) -> (mel (T, n_mels) numpy, durations per phone)."""
    frames = []
    for t in NOISE_TEXTS:
        phones, _ = frontend(t, lang)
        mel, d = synth_mel_and_durations(phones)
        c = np.concatenate([[0], np.cumsum(d)])
        for i, p in enumerate(phones):
            if p == "sp" and d[i] >= 6:
                frames.append(mel[c[i] + 2:c[i + 1] - 2])
    frames = np.concatenate(frames)
    return np.median(frames, axis=0), len(frames)
