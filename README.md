# FastSpeech 2 for English (Svarah) and Telugu (IndicTTS)

Two separately trained FastSpeech 2 text-to-speech models, built on
[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) (commit `d4e79eb`), with
HiFi-GAN (universal) as the vocoder:

| | English | Telugu |
|---|---|---|
| Data | [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) (Indian-accented English, 110 train speakers) | [IndicTTS Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu) (2 speakers) |
| Train / validation | 4.87 h / 0.98 h (3077 / 742 utterances) | 4.96 h / 1.02 h selected; 2140 / 446 utterances after alignment |
| Text front end | lexicon + g2p_en, homograph POS rule, time/number normalisation | rule-based Telugu G2P, Telugu numbers/ordinals/times, English loanwords in Latin script |
| Training | 17,949 steps, 1 × T4, 3.5 h | 55,524 steps, 2 × T4, 8 h |
| Test voice | `svarah_8a8e2655` (clearest recordings, report §5.0); GV post-filter at inference (§5.0b) | `te_female` (better HiFi-GAN copy-synthesis) |

The full write-up is [`Report/Technical_Report.pdf`](Report/Technical_Report.pdf). Every change to the
reference code is listed, with its reason and test, in [`MODIFICATIONS.md`](MODIFICATIONS.md).

## Results

Generated speech for the ten test sentences, with duration, pitch and energy plots of the variance
adaptor, is in [`Results/`](Results), in the layout the assignment asks for:

```
Results/<English|Telugu>/
  training_loss.png  validation_loss.png  loss_curves.png (both, for over-fitting)
  test_generated_audios/<ID>_<lang>.wav
  variance_adaptor_plots/{duration,pitch,energy}/<ID>_<kind>.png
  variance_raw/<ID>.json|.npz          the plotted numbers (per-phone durations, Hz, energy)
  frontend_trace.json                   text -> normalised text -> phones, word by word
  inference_report.json                 consistency checks and timing per sentence
  losses.csv  val_diagnostics.csv  train_config_used.yaml  stats.json  speakers.json
```

### Objective evaluation (held-out validation set; beyond the required outputs)

Computed by [`fastspeech2/evaluate_val.py`](fastspeech2/evaluate_val.py) on every validation utterance,
each with its own speaker, on Kaggle (jobs `fs2-04-eval`, `fs2-06-eval-fix`). Definitions are in
[`fastspeech2/eval_metrics.py`](fastspeech2/eval_metrics.py). Silence phones are excluded from the
duration, pitch and energy metrics.

| Metric | English | Telugu |
|---|---|---|
| Duration MAE (frames of 11.6 ms) / Pearson r of log durations | 2.81 / 0.58 | 1.06 / 0.85 |
| Total-length relative error | 11.7 % | 4.5 % |
| Phone-level pitch RMSE / r | 34.6 Hz / 0.35 | 29.2 Hz / 0.76 |
| Phone-level energy r | 0.78 | 0.90 |
| Mel-cepstral distortion\*, ground-truth variances → predicted variances (DTW) | 36.6 → 39.0 | 26.1 → 30.9 |
| Whisper large-v3 CER, our synthesis vs. copy-synthesis ceiling (150 utterances) | 15.7 % vs 11.6 % | 35.8 % vs 33.1 %† |
| Real-time factor, T4 GPU (acoustic + vocoder, batch 1) | 0.027 (37× faster than real time) | 0.025 (39×) |
| Real-time factor, CPU 4 threads | 0.35 (2.9×) | 0.38 (2.6×) |

\* Computed from the 80-band log-mel (DCT, c1–c24). Not comparable with WORLD/SPTK MCD values
reported in papers; comparable only between rows and languages here.
Copy-synthesis = the ground-truth mel through the same HiFi-GAN, i.e. the best this vocoder and this
ASR model can do; the gap is what the acoustic model loses.
† Whisper's Telugu recognition is weak (33 % CER even on copy-synthesis), so the Telugu figure mainly bounds
the gap (2.7 points) rather than measuring pronunciation.

Extra stress sentences (numbers, questions, homographs, names, abbreviations, long sentences,
Telugu–English code-mixing) are in [`Evaluation/stress_sentences.json`](Evaluation/stress_sentences.json);
their audio, Whisper transcripts and a control-factor check are in `Evaluation/`.

## Checkpoints

Model weights (no optimizer state), one per language, attached to the
[v1.0 release](https://github.com/ashdev-7/iiith-fastspeech2/releases/tag/v1.0) because they exceed
GitHub's 100 MB file limit (checksums in `English/checkpoints/README.md`, `Telugu/checkpoints/README.md`):

| | File | Step |
|---|---|---|
| English | [`en_model.pth.tar`](https://github.com/ashdev-7/iiith-fastspeech2/releases/download/v1.0/en_model.pth.tar) (135 MB) | 17,949 |
| Telugu | [`te_model.pth.tar`](https://github.com/ashdev-7/iiith-fastspeech2/releases/download/v1.0/te_model.pth.tar) (135 MB) | 55,524 |

Load with `torch.load(path, map_location=..., weights_only=True)["model"]`; place them in
`English/checkpoints/` and `Telugu/checkpoints/`.

**Synthesis without the datasets.** Inference needs only the checkpoint plus `stats.json` and
`speakers.json`, which are in `Results/<Language>/`:

```bash
mkdir -p D/preprocessed/en && cp Results/English/stats.json Results/English/speakers.json D/preprocessed/en/
python English/inference.py --data_root D --checkpoint English/checkpoints/en_model.pth.tar \
    --speaker svarah_8a8e2655 --gv_ratio Results/English/gv_ratio.json --gv_strength 1.0 \n    --out_dir my_output          # optional: --pitch_control 1.2 --duration_control 0.9
```

## Reproducing

All compute ran as Kaggle script jobs that clone this repository; [`kaggle/README.md`](kaggle/README.md)
lists them. The same steps run anywhere with Python ≥ 3.10, `pip install -r requirements.txt` and
[Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/) 3.4 (conda-forge).
`D` below is a data root; all paths in the configs are relative to it.

```bash
# 1. Data: download (needs a Hugging Face token with access to both datasets), resample to 22.05 kHz,
#    then a deterministic, leakage-checked 5 h / 1 h split
cd fastspeech2
python -m data.ingest --lang te --cache /tmp/hf --out D        # likewise --lang en
python -m data.select_split --manifest_dir D/manifests/te --train_hours 5 --val_hours 1

# 2. Alignment (MFA): pronunciation dictionary from the project front ends, then
python data/mfa_prepare.py --lang te --ingest_dir D --corpus /tmp/corpus_te --dict_out D/mfa_te/dictionary.txt
mfa train /tmp/corpus_te D/mfa_te/dictionary.txt D/mfa_te/telugu_acoustic_model.zip --output_directory D/preprocessed/te/TextGrid
#    English uses the pretrained english_us_arpa acoustic model:
#    mfa align /tmp/corpus_en D/mfa_en/dictionary.txt english_us_arpa D/preprocessed/en/TextGrid --beam 100 --retry_beam 400
cd ..

# 3. Features (mel, phone-level pitch/energy, durations)
python Telugu/preprocess.py --data_root D

# 4. Training (step counts in the config are the ones used)
python Telugu/train.py --data_root D --max_minutes 480

# 5. Test sentences: wavs, variance plots, variance JSON
python Telugu/inference.py --data_root D --checkpoint Telugu/checkpoints/te_model.pth.tar \
    --speaker te_female --out_dir Results/Telugu

# 6. Optional: objective evaluation
python fastspeech2/evaluate_val.py --config Telugu/config.yaml --data_root D \
    --checkpoint Telugu/checkpoints/te_model.pth.tar --speaker te_female --out_dir Evaluation/Telugu
```

Tests: `pytest` (tests that need torch, g2p_en or MFA are skipped where those are missing; every
Kaggle job runs the full suite before doing anything else).

## Repository layout

```
English/, Telugu/     config.yaml (one per language) + train.py / preprocess.py / inference.py wrappers
fastspeech2/          the modified reference code, Telugu and English front ends, data preparation,
                      inference, plotting, evaluation
kaggle/               the Kaggle job definitions and their shared stages (pipeline.py)
tests/                unit and smoke tests; test_sentences.json = the ten test sentences, checksummed
Results/, Evaluation/ outputs (above)
Report/               Technical_Report.pdf
```

## Data and licences

No dataset audio or text is stored in this repository. Svarah and IndicTTS Telugu are used under the
terms on their dataset cards; the released model weights are trained on them and are provided for
evaluation of this assignment.
