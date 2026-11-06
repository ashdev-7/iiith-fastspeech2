# Kaggle jobs

All compute runs as Kaggle script kernels pushed with the Kaggle CLI. Each job clones this repository,
records the commit in `code_commit.txt`, and calls the shared stage functions in `pipeline.py`.

| Job | Accelerator | Input | Does |
|-----|-------------|-------|------|
| `01_ingest` (`fs2-01-ingest`) | none needed | HF datasets (needs the `HF_TOKEN` Kaggle secret) | inventory, 22.05 kHz wav/.lab, train/val split |
| `00_rehearsal` (`fs2-00-rehearsal`) | CPU | 01 | every later stage end to end on a small subset (tests, MFA, features, copy-synthesis, train + resume, inference, plots) |
| `02_data` (`fs2-02-data`) | CPU | 01 | full MFA alignment, features, copy-synthesis, test-voice choice |
| `03_train_te`, `03_train_en` (`fs2-03-train-<lang>`) | GPU T4 (`--accelerator NvidiaTeslaT4`) | 02 | tests, 300-step throughput probe, dry run of loss plots + inference on the probe checkpoint (fail fast), training sized to an 8 h budget, checkpoints, `training_loss.png` / `validation_loss.png`, test-sentence synthesis + variance plots with RTF timing |
| `04_eval_smoke` (`fs2-04-eval-smoke`) | CPU | 02 | the evaluation code on a 150-step model, 20 utterances, Whisper tiny: finds bugs before the GPU run (numbers meaningless) |
| `04_eval_full` (`fs2-04-eval`) | GPU T4 | 02, 03 te, 03 en | `fastspeech2/evaluate_val.py` for both languages: validation-set metrics, Whisper intelligibility, stress sentences, control check, GPU/CPU RTF |
| `05_reinfer_en` (`fs2-05-reinfer-en`) | CPU | 02, 03 en | English test outputs regenerated with the same checkpoint: v1/v2 after the frontend fixes (rhythm, andaroo), v3 with the clearest-recorded voice `svarah_8a8e2655` and letter-name durations; sentence and stress evaluation with Whisper |
| `06_eval_fix` (`fs2-06-eval-fix`) | GPU T4 | 02, 03 te, 03 en | Telugu evaluation re-run after the Whisper out-of-memory fix |
| `07_diagnose` (`fs2-07-diagnose`) | CPU | 01, 02, 03 te, 03 en | natural vs copy-synthesis vs oracle vs free-running (Praat, band spectrum); clarity ranking of all training speakers; test sentences in the four clearest English voices |
| `08_postfilter` (`fs2-08-postfilter`) | CPU | 02, 03 te, 03 en | GV post-filter: ratio estimated on training clips of each test voice; strengths 0 / 0.5 / 1.0 scored on held-out clips; test sentences rendered at 0.5 and 1.0 |
| `09_vocoder_ft_en`, `09_vocoder_ft_te` (`fs2-09-vocoder-ft-<lang>`) | GPU T4 | 01, 02, 03 | HiFi-GAN fine-tuning on ground-truth-aligned mels (smoke run first, watchdog), then test sentences, diagnosis and full evaluation with the fine-tuned vocoder (not adopted, report §5.0b) |
| `10_combos_en` (`fs2-10-combos-en`) | CPU | 02, 03 en, 09 en | Whisper scores of the final English configuration (universal vocoder + GV post-filter) on the test and stress sentences |
| `11_adapt_en` (`fs2-11-adapt-en`) | CPU | 02, 03 en | Speaker adaptation of the English decoder to the test voice on its training clips only (`fastspeech2/adapt_speaker.py`), early-stopped on its validation clips; post-filter re-estimated; test and stress sentences rendered and Whisper-scored |
| `12_prosody_en` (`fs2-12-prosody-en`) | CPU | 02, 03 en | Lexical-stress realisation in the English data and the model (`fastspeech2/stress_stats.py`); test and stress sentences with the prosody rules (`fastspeech2/prosody_rules.py`) off / terminal fall / fall + stress 0.5 / fall + stress 1.0, rendered and Whisper-scored |
| `13_data_accounting` (`fs2-13-data-accounting`) | CPU | 01, 02 | Selected vs used utterances and hours per split, after alignment and preprocessing; train/val overlap of the used sets (`Evaluation/Data/accounting.json`) |
| `14_extended` (`fs2-14-extended`) | CPU | 02, 03 en, 03 te | Extended evaluation set (`Evaluation/Extended/extended_sentences.json`, 10 + 10 sentences beyond the assignment): renders with plots and Whisper scores, with and without the prosody rules |
| `15_stress_stats` (`fs2-15-stress-stats`) | CPU | 02, 03 en | Re-run of job 12's stress-realisation measurement (job 12 failed writing its JSON) |
| `16_clarity` (`fs2-16-clarity`) | CPU | 02, 03 en, 03 te | Clear-speech options (`fastspeech2/clarity.py`, phrasing rule): English variants and Telugu fall / edge-fade variants rendered and Whisper-scored |
| `17_final_en` (`fs2-17-final-en`) | CPU | 02, 03 en | Final English configuration: `Results/English` and the extended English set rendered and Whisper-scored |
| `18_aligner_check` (`fs2-18-aligner-check`) | CPU | 01, 02, 03 te | Telugu MFA aligner retrained on the training clips only; validation clips aligned with it; alignments and the model's validation duration scores compared with the reported ones (`tools/aligner_check.py`) |
| `19_asr_te` (`fs2-19-asr-te`) | CPU | 02, 03 te | Telugu intelligibility re-measured with a Telugu ASR model (`vasista22/whisper-telugu-large-v2`, pinned; not trained on IndicTTS): validation synthesis vs copy-synthesis ceiling, assignment, extended and stress sentences |

`03_*` and `04_*` folders are generated from `03_train/run_template.py` and `04_eval/run_template.py`
by `tools/make_train_jobs.py`; `tests/test_pipeline.py` fails if they are out of date.

Failure-mode guards: the test suite runs first in every job (`pytest_gate`); CUDA is exercised at start-up
on GPU jobs; each stage is timed and its error recorded in `status.json`, which is rewritten after every
stage; non-fatal stages let the other language continue; training stops on non-finite loss and
checkpoints before the session limit (`--max_minutes`), resuming with `--restore_step latest`.

API budget (Kaggle rate limits): a job leaves only a handful of files in `/kaggle/working`
(tars + small JSON), so listing its output is one cheap call. Per job: 1 push, status polls no more
often than every 5 minutes, then one fetch of `status.json` (and at most one small tar). Big tars are
never downloaded; later jobs read them through `kernel_sources`.
