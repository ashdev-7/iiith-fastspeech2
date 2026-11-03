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
| M2 | `audio/stft.py` (`STFT.transform`) | `input.cuda()` … `.cpu()` hard-coded | Crashes on CPU-only machines (feature extraction runs in a Kaggle CPU job) | Uses the device of `forward_basis` | both | Kaggle job `fs2-02-data` (features for 6,405 utterances on CPU) |
| M3 | `audio/stft.py`, `audio/audio_processing.py` | `librosa_mel_fn(sr, n_fft, …)`, `pad_center(x, n)` positional | librosa ≥ 0.10 accepts these only as keyword arguments | Keyword arguments | both | Kaggle job `fs2-02-data` (librosa 0.11) |
| M4 | `dataset.py` | `speakers.json` opened without `encoding` | Windows default code page is not UTF-8 | `encoding="utf-8"` | both | Code review |
| M5 | `preprocessor/preprocessor.py` (`build_from_path`) | Utterances without a TextGrid still added the *previous* utterance's pitch/energy to the scalers and frame count (variables left over from the last loop iteration) | Bug: biased normalisation statistics | Statistics are updated only for utterances that were processed | both | `tests/test_preprocessor.py` (Kaggle) |
| M6 | `preprocessor/preprocessor.py` (`process_utterance`, `get_alignment`) | `tgt.io.read_textgrid(path)` drops empty intervals | MFA ≥ 2 writes silences as empty intervals, so inter-word pauses vanished from the phone sequence and durations | `include_empty_intervals=True`; empty labels mapped to `sp` (leading/trailing silence is still trimmed) | both | `tests/test_preprocessor.py` (Kaggle) |
| M7 | `preprocessor/preprocessor.py` | `librosa.load(path)` (implicit 22050 Hz), `.lab` opened without `encoding`, `os.listdir` order for speaker ids | Explicit rate from config; Telugu text on non-UTF-8 platforms; reproducible speaker ids | `sr=sampling_rate`, `encoding="utf-8"`, speakers sorted | both | `tests/test_preprocessor.py` (Kaggle) |
| M8 | `utils/model.py` | `torch.load` without `map_location`; HiFi-GAN paths relative to the working directory; checkpoint must be unzipped by hand | CPU inference; scripts run from `English/`, `Telugu/`; torch ≥ 2.6 `weights_only` default | `map_location=device, weights_only=True`; paths resolved from the package; `.zip` extracted on first use | both | Kaggle jobs `fs2-02-data` (copy-synthesis), `fs2-03-train-*` (inference) |
| M9 | `utils/tools.py` (`to_device`) | Energy targets moved to the device without a dtype cast | Normalised energy `.npy` files are float64 (divided by float64 statistics), so the energy MSE was double vs float and `backward()` failed ("Found dtype Double but expected Float", found by the Kaggle rehearsal) | `.float()`, as already done for pitch | both | `tests/test_model_smoke.py::test_to_device_casts_float64_features` |
| M10 | `model/optimizer.py` (`ScheduledOptim._update_learning_rate`) | Learning rate computed with `np.power` stored as a numpy scalar in the optimizer state | The numpy scalar is pickled into checkpoints; torch >= 2.6 `torch.load(weights_only=True)` rejects it, so resume and inference failed (found by the Kaggle rehearsal) | `float(...)` | both | `tests/test_model_smoke.py::test_train_checkpoint_loads_with_weights_only` |
| M11 | `utils/tools.py` (`get_mask_from_lengths`), `model/modules.py` (`LengthRegulator.LR`) | Masks and mel lengths created on the module-level global `device` (CUDA whenever a GPU exists) | A model copy on the CPU fails on a GPU machine (device mismatch), which the CPU speed measurement needs | Use the device of the input tensor; identical behaviour on a single device | both | Kaggle job `fs2-04-eval-smoke` / `fs2-04-eval` |
| M12 | `model/modules.py` (`VarianceAdaptor.get_pitch_embedding`, `get_energy_embedding`) | `p_control` / `e_control` multiplied the z-normalised prediction | That scales the distance from the corpus mean, not the value: for a voice below the mean, pitch ×1.2 lowered F0 (English test voice: 126 → 113 Hz; ×0.8 → 140 Hz, measured in `fs2-05-reinfer-en`) | Scale in physical units: `((p·std+mean)·c − mean)/std`; control 1.0 (training, default inference) unchanged | both | `tests/test_model_smoke.py::test_pitch_energy_controls_scale_physical_units`; `fs2-06-eval-fix` control check |

## Language support

| # | File | Original behaviour | Reason | New behaviour | Lang | Validation |
|---|------|--------------------|--------|---------------|------|------------|
| L1 | `text/symbols.py` | One global table: ASCII letters + ARPAbet + pinyin | No Telugu symbols; vocabulary must match each language's frontend | Adds a Telugu table (padding, punctuation, `@`-prefixed Telugu phones, silences) and `get_symbols(language)`; the English table is unchanged | te | `tests/test_symbols.py` |
| L2 | `text/__init__.py` | `text_to_sequence(text, cleaners)` with the global table; unknown symbols silently dropped | Per-language ids; silent dropping hides frontend/vocabulary mismatches | `text_to_sequence(text, cleaners, language="en")`, `sequence_to_text(seq, language="en")`; unknown symbols raise `ValueError` | both | `tests/test_symbols.py` |
| L3 | `transformer/Models.py` (`Encoder`) | `n_src_vocab = len(symbols) + 1` | Embedding size must follow the language's table | `len(get_symbols(config["text_language"])) + 1` | both | `tests/test_model_smoke.py` (Kaggle; embedding size checked for both languages) |
| L4 | `dataset.py` (`Dataset`, `TextDataset`) | `text_to_sequence(text, cleaners)` | Pass the language | Reads `preprocessing.text.language`, passes it on | both | `tests/test_model_smoke.py`; training jobs |

## New files (not in the reference)

| File | Purpose | Validation |
|------|---------|------------|
| `text/telugu/phones.py`, `g2p.py`, `normalize.py` | Rule-based Telugu normalisation (numbers, ordinals, times) and grapheme-to-phone conversion; raises on any unconvertible character | `tests/test_telugu_frontend.py` |
| `data/ingest.py` | HF parquet → inventory + 22.05 kHz wav/.lab (Kaggle) | Kaggle job `fs2-01-ingest` |
| `data/select_split.py` | Deterministic, duration-targeted train/val split with leakage checks | `tests/test_split.py` |
| `utils/config.py`, `English/config.yaml`, `Telugu/config.yaml` | One config file per language (preprocess / model / train sections, paths relative to a data root); injects `text_language` into the model config; `max_seq_len` 1500 because clips reach 15 s (1292 frames) | Loaded for both languages locally |
| `train.py`, `preprocess.py`, `evaluate.py` (CLI) | Training loop from the reference `train.py` with one config per language, fixed seeds, CSV loss log (`losses.csv`), `--max_minutes` clean stop + checkpoint, `--restore_step latest`, atomic checkpoint writes; `evaluate()` can return the loss means | `tests/test_model_smoke.py` (Kaggle), job 03 smoke run |
| `text/english_frontend.py` | English inference frontend: time/number normalisation, g2p_en homograph rule, reference lexicon, g2p_en for OOV words; trace of every word. Added after the first full English run: split-word joining for pronunciation (E3 "rhyth m" had been read "R AY1 DH" + letter M) and word-final "oo" → UW for OOV words (E4 "andaroo" had ended in OW); the test text itself is unchanged. Outputs regenerated by job `fs2-05-reinfer-en` with the same checkpoint | `tests/test_english_frontend.py` |
| `inference.py`, `plotting.py` | Test-sentence synthesis, variance export (JSON/NPZ), duration/pitch/energy plots, loss plots, runtime consistency checks | `tests/test_model_smoke.py` (Kaggle) |
| `copy_synthesis.py` | HiFi-GAN copy-synthesis distortion per speaker (test-voice choice for Telugu) | Kaggle job `fs2-02-data` |
| `English/*.py`, `Telugu/*.py` | Thin wrappers that run the shared scripts with the folder's `config.yaml` | — |
| `tools/build_test_manifest.py` | Canonical test sentences from the assignment PDF text layer | `tests/test_test_manifest.py` |
| `kaggle/*/run.py`, `kernel-metadata.json` | Kaggle job definitions; each job clones this repository and records the commit in `code_commit.txt` | — |
| `text/telugu/latin.py` | English words in Latin script inside Telugu text (code-mixing): English pronunciation → ARPAbet → nearest Telugu phones (only phones already in the Telugu symbol table; training data contains no Latin script, so features and test sentences are unchanged). Before: G2PError | `tests/test_telugu_frontend.py` |
| `evaluate_val.py`, `eval_metrics.py` | Objective evaluation beyond the required outputs: validation-set duration/pitch/energy errors, mel-cepstral distortion (oracle vs predicted variances), Whisper intelligibility vs a copy-synthesis ceiling, stress sentences (`Evaluation/stress_sentences.json`), control-factor check, GPU/CPU real-time factor | `tests/test_eval_metrics.py`; Kaggle job `fs2-04-eval-smoke` |
| `tools/acoustic_analysis.py`, `diagnose_quality.py`, `tools/word_stats.py` | Praat analysis per sentence and per word (HNR, high-band share, centre of gravity, jitter, shimmer, F0, rate; flags noisy/dull/rushed words); separation of vocoder, decoder and variance-predictor effects against natural recordings; clarity ranking of training speakers (used to choose the English test voice); per-word duration/pitch numbers behind the report's §6 | `tests/test_acoustic_analysis.py`; Kaggle job `fs2-07-diagnose` |
| `text/english_frontend.py` (letter names), `inference.py`, `evaluate_val.py` | Letter names (AM/PM) get per-phone duration scale 1.6, passed to the variance adaptor as a (1, L) `d_control`; AM had been predicted in 244 ms and heard as "pm" | `tests/test_english_frontend.py`, `tests/test_model_smoke.py::test_per_phone_duration_control`; job `fs2-05-reinfer-en` v3 |

## Deliberate deviations from the reference conventions

| Item | Reference | This project | Reason |
|------|-----------|--------------|--------|
| Peak normalisation in `prepare_align` | `wav / max|wav| * 32768.0` | `* 32767.0` (in `data/ingest.py`) | 32768 overflows int16 at the peak sample |
| Train/val split | `random.shuffle` without seed, fixed `val_size` count | Manifest-based, seeded, duration-targeted (`data/select_split.py`); `preprocessor.py` reads `path.split_path`/{train,val}.tsv and only processes listed utterances | Reproducibility; the assignment's 5 h / 1 h targets; no overlap |
