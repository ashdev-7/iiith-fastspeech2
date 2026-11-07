# FastSpeech 2 for English (Svarah) and Telugu (IndicTTS)

Two separately trained FastSpeech 2 text-to-speech models, built on
[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) (commit `d4e79eb`), with
HiFi-GAN (universal) as the vocoder:

| | English | Telugu |
|---|---|---|
| Data | [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) (Indian-accented English; 110 speaker profiles, as Svarah has no speaker ids) | [IndicTTS Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu) (2 speakers) |
| Train / validation | 4.87 h / 0.98 h (3077 / 742 utterances) | 4.90 h / 1.01 h (2140 / 446 utterances; 4.96 h / 1.02 h selected, 28 clips not aligned) |
| Text front end | lexicon + g2p_en, homograph POS rule, time/number normalisation | rule-based Telugu G2P, Telugu numbers/ordinals/times, English loanwords in Latin script |
| Training | 17,949 steps, 1 × T4, 3.5 h | 55,524 steps, 2 × T4, 8 h |
| Test voice | `svarah_8a8e2655` (clearest recordings, report Section 6.2) | `te_female` (better HiFi-GAN copy-synthesis) |
| Inference | GV post-filter and clear-speech options (report Sections 6.3–6.4); model weights unchanged | the model as trained, no options |

The full write-up is [`Report/Technical_Report.pdf`](Report/Technical_Report.pdf). Every change to the
reference code is listed, with its reason and test, in [`MODIFICATIONS.md`](MODIFICATIONS.md), which starts
with a one-table summary of the Telugu support.

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
each with its own speaker and without inference-time options, on Kaggle (jobs `fs2-04-eval`,
`fs2-06-eval-fix`). Definitions are in
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
| Real-time factor, T4 GPU (acoustic + vocoder, batch 1) | 0.027 (37× faster than real time) | 0.026 (39×) |
| Real-time factor, CPU 4 threads | 0.35 (2.9×) | 0.37 (2.7×) |

\* Computed from the 80-band log-mel (DCT, c1–c24). Not comparable with WORLD/SPTK MCD values
reported in papers; comparable only between rows and languages here.
Copy-synthesis = the ground-truth mel through the same HiFi-GAN, i.e. the best this vocoder and this
ASR model can do; the gap is what the acoustic model loses.
† Whisper large-v3 is weak on Telugu (33 % CER even on copy-synthesis of real recordings), so the Telugu
figure bounds the gap (2.7 points) rather than measuring pronunciation; it also writes some Telugu speech in
Kannada script, which the report scores after mapping it back to Telugu (Section 6.6, Section 8.4).

Further sentences beyond the assignment (not part of it): 16 stress sentences written for this project (numbers, questions, homographs, names,
abbreviations, long sentences, Telugu–English code-mixing; [`Evaluation/stress_sentences.json`](Evaluation/stress_sentences.json))
and 20 extended sentences supplied by the author ([`Evaluation/Extended/extended_sentences.json`](Evaluation/Extended/extended_sentences.json); XE01 and XT01 repeat E1 and T1 verbatim),
scored in the report (Sections 8.3–8.4). `Evaluation/English/model_only/` holds the English test sentences from the
model alone (no inference-time options), for comparison with `Results/English/`.

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
    --speaker svarah_8a8e2655 --out_dir my_output \
    --gv_ratio Results/English/gv_ratio.json --gv_strength 1.0 \
    --prosody_fall --prosody_stress 1.0 --prosody_phrasing --duration_control 1.10 \
    --long_word_scale 1.25 --heavy_word_pause \
    --denoise_profile Results/English/noise_profile.json --edge_fade
mkdir -p D/preprocessed/te && cp Results/Telugu/stats.json Results/Telugu/speakers.json D/preprocessed/te/
python Telugu/inference.py --data_root D --checkpoint Telugu/checkpoints/te_model.pth.tar \
    --speaker te_female --out_dir my_output_te
```

`--sentences file.json` synthesises other text (same format as `tests/test_sentences.json`, with each
sentence's SHA-256). Leaving out the English options gives the model-only output.

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
python -m data.select_split --manifest_dir D/manifests/te --train_hours 5 --val_hours 1 --balance equal   # English: proportional (default)

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

# English: steps 3-5 (after step 1 with --lang en and the pretrained-aligner command in step 2)
python English/preprocess.py --data_root D
python English/train.py --data_root D --max_minutes 210          # 17,949 steps in the reported run
python tools/noise_profile.py --config English/config.yaml --data_root D \
    --checkpoint English/checkpoints/en_model.pth.tar --speaker svarah_8a8e2655 \
    --gv_ratio Results/English/gv_ratio.json --gv_strength 1.0 --out Results/English/noise_profile.json
python English/inference.py --data_root D --checkpoint English/checkpoints/en_model.pth.tar \
    --speaker svarah_8a8e2655 --out_dir Results/English \
    --gv_ratio Results/English/gv_ratio.json --gv_strength 1.0 \
    --prosody_fall --prosody_stress 1.0 --prosody_phrasing --duration_control 1.10 \
    --long_word_scale 1.25 --heavy_word_pause \
    --denoise_profile Results/English/noise_profile.json --edge_fade
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
tools/                report build, measurement and analysis scripts
Results/, Evaluation/ outputs (above)
Report/               Technical_Report.pdf (source: Report/src/report.html, built by tools/build_report.py)
```

## Data and licences

No dataset audio or text is stored in this repository. Svarah and IndicTTS Telugu are used under the
terms on their dataset cards; the released model weights are trained on them and are provided for
evaluation of this assignment.
