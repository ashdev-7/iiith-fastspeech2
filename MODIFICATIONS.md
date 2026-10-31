# Modifications to the reference implementation

Reference: [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2), commit
`d4e79eb52e8b01d24703b2dfc0385544092958f3` (2021-07-08), MIT License (© 2020 Chung-Ming Chien).
The reference code is vendored in `fastspeech2/`; its `LICENSE` and `hifigan/LICENSE` are kept.

Vendored: `audio/`, `model/`, `transformer/`, `text/`, `utils/`, `dataset.py`, `evaluate.py`,
`preprocessor/preprocessor.py`, `hifigan/` (code, config, LICENSE, `generator_universal.pth.tar.zip`),
`lexicon/librispeech-lexicon.txt`. Not vendored: the LJSpeech/AISHELL3/LibriTTS configs and
preprocessed metadata, `hifigan/generator_LJSpeech.pth.tar.zip`, the pinyin lexicon.

Each entry lists: file · original behaviour · reason · new behaviour · languages · validation.

## Bug fixes and portability

| # | File | Original behaviour | Reason | New behaviour | Lang | Validation |
|---|------|--------------------|--------|---------------|------|------------|
| M1 | `model/modules.py` (`VarianceAdaptor.forward`, 2 places) | Energy embedding called with `p_control` | Bug: energy control used the pitch control value | Uses `e_control` | both | Code review; affects only control values ≠ 1.0 |
| M2 | `audio/stft.py` (`STFT.transform`) | `input.cuda()` … `.cpu()` hard-coded | Crashes on CPU-only machines (feature extraction runs in a Kaggle CPU job) | Uses the device of `forward_basis` | both | Pending: feature-extraction job |
| M3 | `audio/stft.py`, `audio/audio_processing.py` | `librosa_mel_fn(sr, n_fft, …)`, `pad_center(x, n)` positional | librosa ≥ 0.10 accepts these only as keyword arguments | Keyword arguments | both | Pending: feature-extraction job |
| M4 | `dataset.py` | `speakers.json` opened without `encoding` | Windows default code page is not UTF-8 | `encoding="utf-8"` | both | Code review |

## Language support

| # | File | Original behaviour | Reason | New behaviour | Lang | Validation |
|---|------|--------------------|--------|---------------|------|------------|
| L1 | `text/symbols.py` | One global table: ASCII letters + ARPAbet + pinyin | No Telugu symbols; vocabulary must match each language's frontend | Adds a Telugu table (padding, punctuation, `@`-prefixed Telugu phones, silences) and `get_symbols(language)`; the English table is unchanged | te | `tests/test_symbols.py` |
| L2 | `text/__init__.py` | `text_to_sequence(text, cleaners)` with the global table; unknown symbols silently dropped | Per-language ids; silent dropping hides frontend/vocabulary mismatches | `text_to_sequence(text, cleaners, language="en")`, `sequence_to_text(seq, language="en")`; unknown symbols raise `ValueError` | both | `tests/test_symbols.py` |
| L3 | `transformer/Models.py` (`Encoder`) | `n_src_vocab = len(symbols) + 1` | Embedding size must follow the language's table | `len(get_symbols(config["text_language"])) + 1` | both | Pending: model smoke test |
| L4 | `dataset.py` (`Dataset`, `TextDataset`) | `text_to_sequence(text, cleaners)` | Pass the language | Reads `preprocessing.text.language`, passes it on | both | Pending: model smoke test |

## New files (not in the reference)

| File | Purpose | Validation |
|------|---------|------------|
| `text/telugu/phones.py`, `g2p.py`, `normalize.py` | Rule-based Telugu normalisation (numbers, ordinals, times) and grapheme-to-phone conversion; raises on any unconvertible character | `tests/test_telugu_frontend.py` |
| `data/ingest.py` | HF parquet → inventory + 22.05 kHz wav/.lab (Kaggle) | Kaggle job `fs2-01-ingest` |
| `data/select_split.py` | Deterministic, duration-targeted train/val split with leakage checks | `tests/test_split.py` |
| `tools/build_test_manifest.py` | Canonical test sentences from the assignment PDF text layer | `tests/test_test_manifest.py` |
| `tools/stage_kaggle_code.py`, `kaggle/*` | Kaggle code dataset and job definitions | — |

## Deliberate deviations from the reference conventions

| Item | Reference | This project | Reason |
|------|-----------|--------------|--------|
| Peak normalisation in `prepare_align` | `wav / max|wav| * 32768.0` | `* 32767.0` (in `data/ingest.py`) | 32768 overflows int16 at the peak sample |
| Train/val split | `random.shuffle` without seed, fixed `val_size` count | Manifest-based, seeded, duration-targeted (`data/select_split.py`) | Reproducibility; the assignment's 5 h / 1 h targets; no overlap |
