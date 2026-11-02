"""Objective evaluation of a trained model (beyond the assignment's required outputs; nothing here
changes the training data, the model or the required Results/).

  python evaluate_val.py --config ../Telugu/config.yaml --data_root D --checkpoint te_model.pth.tar \
      --speaker te_female --out_dir ../Evaluation/Telugu

1. Held-out validation set (never trained on), each utterance with its own speaker:
   * free-running synthesis (predicted duration/pitch/energy):
       duration  : MAE in frames and Pearson r of log(1+d) against the MFA durations
       pitch     : phone-level RMSE (Hz) and Pearson r against the phone-averaged ground-truth F0
       energy    : phone-level RMSE and Pearson r
       length    : relative error of the total predicted length
       MCD-mel   : DTW-aligned mel-cepstral distortion against the ground-truth mel
   * oracle synthesis (ground-truth duration/pitch/energy fed in): MCD-mel, frame-aligned. The gap
     between the two isolates what the variance predictors cost.
   Silence phones (sp/spn/sil) are excluded from the duration/pitch/energy metrics.
2. Intelligibility on a fixed subset: Whisper transcribes (a) our free-running synthesis and
   (b) copy-synthesis (ground-truth mel -> the same HiFi-GAN). (b) is the ceiling for this vocoder
   and this ASR model, so the gap (a) - (b) is what the acoustic model loses. CER/WER are corpus level.
3. The assignment's test sentences and extra stress sentences (Evaluation/stress_sentences.json),
   synthesised with the test voice and transcribed by Whisper (reported for reading; the reference is
   the frontend's normalised text, so "10:45" is compared as words).
4. Control check: duration/pitch/energy control factors, measured on the generated audio.
5. Speed: real-time factor (RTF = compute seconds / audio seconds), batch 1, warm, on the GPU and on
   the CPU, acoustic model and vocoder separately.
Copy-synthesis audio is derived from the dataset and is never written to disk.
"""
import argparse
import copy
import csv
import json
import os
import time

import numpy as np
import torch
from scipy.io import wavfile

import eval_metrics as M
from dataset import Dataset
from inference import _now, device, frontend, load_model
from text import text_to_sequence
from utils.config import load_configs
from utils.model import get_vocoder, vocoder_infer

SILENCES = {"sp", "spn", "sil"}
ASR_LANGUAGE = {"en": "english", "te": "telugu"}
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def _t(x, dtype=torch.float32):
    return torch.from_numpy(np.asarray(x)[None]).to(device=device, dtype=dtype)


def synthesize(model, vocoder, configs, speaker_id, seq, controls=(1.0, 1.0, 1.0), dev=None):
    """Free-running synthesis. Returns dict with mel (T, 80), wav int16, variances and timings."""
    dev = dev or device
    pre_cfg, model_cfg, _ = configs
    texts = torch.from_numpy(np.array([seq])).long().to(dev)
    src_lens = torch.tensor([len(seq)], device=dev)
    spk = torch.tensor([speaker_id], device=dev).long()
    t0 = _now(dev)
    with torch.no_grad():
        out = model(spk, texts, src_lens, len(seq), p_control=controls[0], e_control=controls[1], d_control=controls[2])
    t1 = _now(dev)
    mel_len = int(out[9][0].item())
    res = {"p": out[2][0, :len(seq)].cpu().numpy(), "e": out[3][0, :len(seq)].cpu().numpy(),
           "d": out[5][0, :len(seq)].cpu().numpy().astype(int), "mel_len": mel_len, "acoustic_s": t1 - t0}
    if mel_len == 0:
        res["wav"], res["vocoder_s"] = None, 0.0
        return res
    mel = out[1][0, :mel_len]
    res["mel"] = mel.cpu().numpy()
    t2 = _now(dev)
    res["wav"] = vocoder_infer(mel.transpose(0, 1).unsqueeze(0), vocoder, model_cfg, pre_cfg)[0]
    res["vocoder_s"] = _now(dev) - t2
    return res


def oracle_mel(model, s):
    """Mel with ground-truth duration/pitch/energy (teacher forcing of the variance adaptor)."""
    seq = s["text"]
    T = s["mel"].shape[0]
    with torch.no_grad():
        out = model(torch.tensor([s["speaker"]], device=device).long(), _t(seq, torch.long),
                    torch.tensor([len(seq)], device=device), len(seq), mel_lens=torch.tensor([T], device=device),
                    max_mel_len=T, p_targets=_t(s["pitch"]), e_targets=_t(s["energy"]),
                    d_targets=_t(s["duration"], torch.long))
    return out[1][0, :T].cpu().numpy()


def f0_stats(wav, sr, frame_ms):
    import pyworld as pw
    x = wav.astype(np.float64) / 32768.0
    f0, t = pw.dio(x, sr, frame_period=frame_ms)
    f0 = pw.stonemask(x, f0, t, sr)
    v = f0[f0 > 0]
    return float(np.median(v)) if len(v) else None


def load_asr(name):
    from transformers import pipeline
    kw = {"model": name, "device": 0 if torch.cuda.is_available() else -1, "chunk_length_s": 30}
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        return pipeline("automatic-speech-recognition", dtype=dtype, **kw)
    except TypeError:  # older transformers
        return pipeline("automatic-speech-recognition", torch_dtype=dtype, **kw)


def transcribe(asr, wavs, sr, lang, batch_size=8):
    import librosa
    if not wavs:
        return []
    inputs = [{"raw": librosa.resample(w.astype(np.float32) / 32768.0, orig_sr=sr, target_sr=16000),
               "sampling_rate": 16000} for w in wavs]
    kw = {"generate_kwargs": {"language": ASR_LANGUAGE[lang], "task": "transcribe"}}
    try:
        outs = asr(inputs, batch_size=batch_size, **kw)
    except torch.cuda.OutOfMemoryError:  # Telugu: byte-level tokens make batches of 8 too large (fs2-04-eval)
        torch.cuda.empty_cache()
        outs = [asr(x, **kw) for x in inputs]
    return [o["text"].strip() for o in outs]


def whisper_english_normaliser(asr):
    """Whisper's EnglishTextNormalizer (numbers -> digits, British/American spelling, "a.m."), or None."""
    try:
        f = asr.tokenizer.normalize
        f("Ten forty five a.m.")
        return f
    except Exception:  # noqa: BLE001
        return None


def main(args):
    configs = load_configs(args.config, args.data_root)
    pre_cfg, model_cfg, train_cfg = configs
    lang = pre_cfg["preprocessing"]["text"]["language"]
    pre = pre_cfg["path"]["preprocessed_path"]
    p = pre_cfg["preprocessing"]
    sr, hop = p["audio"]["sampling_rate"], p["stft"]["hop_length"]
    frame_ms = 1000.0 * hop / sr
    stats = json.load(open(os.path.join(pre, "stats.json"), encoding="utf-8"))
    speakers = json.load(open(os.path.join(pre, "speakers.json"), encoding="utf-8"))
    pm, ps = stats["pitch"][2:4]
    em, es = stats["energy"][2:4]
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    report = {"checkpoint": os.path.basename(args.checkpoint), "lang": lang, "device": str(device) + (
        " ({})".format(torch.cuda.get_device_name(0)) if device.type == "cuda" else ""), "definitions": M.__doc__}

    model, step = load_model(configs, args.checkpoint)
    report["checkpoint_step"] = step
    vocoder = get_vocoder(model_cfg, device)
    ds = Dataset("val.txt", pre_cfg, train_cfg, sort=False, drop_last=False)
    n = len(ds) if args.max_utts <= 0 else min(args.max_utts, len(ds))
    if args.sentences_only:  # only parts 3/4 (+ their ASR): e.g. after a frontend change
        n = 0
        report["mode"] = "sentences_only"
    asr_idx = set(np.linspace(0, n - 1, min(args.asr_utts, n)).round().astype(int).tolist()) if args.asr_utts else set()

    # Warm-up (CUDA init kept out of the timings).
    s0 = ds[0]
    synthesize(model, vocoder, configs, s0["speaker"], s0["text"])

    # ---------------------------------------------------------------- 1. validation set
    rows, asr_items = [], []
    t_start = time.time()
    for i in range(n):
        s = ds[i]
        phones = ds.text[i].strip("{}").split()
        keep = np.array([ph not in SILENCES for ph in phones])
        r = synthesize(model, vocoder, configs, s["speaker"], s["text"])
        row = {"id": s["id"], "speaker": ds.speaker[i], "phones": len(phones), "gt_frames": int(s["mel"].shape[0]),
               "pred_frames": r["mel_len"], "acoustic_s": r["acoustic_s"], "vocoder_s": r["vocoder_s"]}
        d_gt, d_pr = s["duration"].astype(float), r["d"].astype(float)
        row["duration_mae_frames"] = float(np.mean(np.abs(d_pr[keep] - d_gt[keep]))) if keep.any() else None
        row["duration_log_r"] = M.pearson(np.log1p(d_pr[keep]), np.log1p(d_gt[keep]))
        row["length_rel_error"] = abs(r["mel_len"] - d_gt.sum()) / max(1.0, d_gt.sum())
        hz_pr, hz_gt = r["p"] * ps + pm, s["pitch"] * ps + pm
        row["pitch_rmse_hz"] = M.rmse(hz_pr[keep], hz_gt[keep])
        row["pitch_r"] = M.pearson(hz_pr[keep], hz_gt[keep])
        en_pr, en_gt = r["e"] * es + em, s["energy"] * es + em
        row["energy_rmse"] = M.rmse(en_pr[keep], en_gt[keep])
        row["energy_r"] = M.pearson(en_pr[keep], en_gt[keep])
        c_gt = M.mel_cepstra(s["mel"])
        c_or = M.mel_cepstra(oracle_mel(model, s))
        row["mcd_mel_oracle"] = M.mcd_aligned(c_or, c_gt)
        # DTW lowers MCD by itself (it picks the cheapest frame pairing), so the oracle is also scored
        # with DTW: compare mcd_mel_oracle_dtw with mcd_mel_free_dtw, never mcd_mel_oracle with it.
        row["mcd_mel_oracle_dtw"] = M.mcd_dtw(c_or, c_gt)[0]
        row["mcd_mel_free_dtw"] = M.mcd_dtw(M.mel_cepstra(r["mel"]), c_gt)[0] if r["mel_len"] else None
        row["audio_s"] = len(r["wav"]) / sr if r["wav"] is not None else 0.0
        if r["wav"] is not None:
            if i in asr_idx and len(asr_items) < args.save_samples:  # a few synthesised samples to listen to
                os.makedirs(os.path.join(out, "samples"), exist_ok=True)
                wavfile.write(os.path.join(out, "samples", "{}_{}.wav".format(ds.speaker[i], s["id"])), sr, r["wav"])
            if i in asr_idx:
                mel_gt = torch.from_numpy(s["mel"].T).float().unsqueeze(0).to(device)
                copy_wav = vocoder_infer(mel_gt, vocoder, model_cfg, pre_cfg)[0]  # in memory only
                asr_items.append({"id": s["id"], "speaker": ds.speaker[i], "ref": ds.raw_text[i],
                                  "synth": r["wav"], "copy": copy_wav})
        rows.append(row)
        if i % 100 == 0:
            print("val {}/{} ({:.1f} min)".format(i, n, (time.time() - t_start) / 60), flush=True)
    if rows:
        with open(os.path.join(out, "val_per_utterance.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), restval="")
            w.writeheader()
            w.writerows(rows)
    keys = ["duration_mae_frames", "duration_log_r", "length_rel_error", "pitch_rmse_hz", "pitch_r", "energy_rmse",
            "energy_r", "mcd_mel_oracle", "mcd_mel_oracle_dtw", "mcd_mel_free_dtw"]
    report["validation"] = {"utterances": n, "speakers": len(set(ds.speaker[:n])),
                            "metrics": {k: M.summarise([r.get(k) for r in rows]) for k in keys},
                            "zero_length_predictions": sum(1 for r in rows if not r["pred_frames"])}
    timed = [r for r in rows if r["audio_s"]]
    audio = sum(r["audio_s"] for r in timed)
    voc = [r for r in timed if r["vocoder_s"]]
    if timed:
        report["speed_gpu" if device.type == "cuda" else "speed_cpu_full"] = {
            "utterances": len(timed), "audio_s": audio,
            "rtf_acoustic": sum(r["acoustic_s"] for r in timed) / audio,
            "rtf_vocoder": sum(r["vocoder_s"] for r in voc) / max(1e-9, sum(r["audio_s"] for r in voc)),
            "note": "batch 1, warm (one untimed warm-up call), CUDA-synchronised; excludes text frontend"}
    json.dump(report, open(os.path.join(out, "metrics.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    # ---------------------------------------------------------------- 3/4. test, stress, controls
    spk_id = speakers[args.speaker]
    sentences = [dict(it, set="assignment") for it in json.load(open(
        os.path.join(REPO, "tests", "test_sentences.json"), encoding="utf-8"))["items"] if it["lang"] == lang]
    stress_path = os.path.join(REPO, "Evaluation", "stress_sentences.json")
    if os.path.exists(stress_path):
        sentences += [dict(it, set="stress") for it in json.load(open(stress_path, encoding="utf-8"))["items"]
                      if it["lang"] == lang]
    os.makedirs(os.path.join(out, "stress"), exist_ok=True)
    sent_rows = []
    for it in sentences:
        text = it.get("canonical") or it["text"]
        rec = {"id": it["id"], "set": it["set"], "category": it.get("category"), "text": text}
        try:
            phones, trace = frontend(text, lang)
            seq = text_to_sequence("{" + " ".join(phones) + "}", [], lang)
            r = synthesize(model, vocoder, configs, spk_id, seq)
            rec.update({"model_input": trace["model_input"], "phones": " ".join(phones), "audio_s": (
                len(r["wav"]) / sr) if r["wav"] is not None else 0, "wav": r["wav"]})
            if it["set"] == "stress" and r["wav"] is not None:
                wavfile.write(os.path.join(out, "stress", "{}.wav".format(it["id"])), sr, r["wav"])
        except Exception as e:  # noqa: BLE001  (a frontend failure on a stress sentence is a finding)
            rec["error"] = "{}: {}".format(type(e).__name__, e)
        sent_rows.append(rec)

    controls_text = {"en": "The researchers will conduct an experiment to evaluate the proposed method.",
                     "te": "తాడుపై అతని పట్టు చాలా బలంగా ఉంది."}[lang]
    phones, _ = frontend(controls_text, lang)
    seq = text_to_sequence("{" + " ".join(phones) + "}", [], lang)
    os.makedirs(os.path.join(out, "controls"), exist_ok=True)
    ctrl = []
    for name, c in [("base", (1.0, 1.0, 1.0)), ("duration_0.8", (1.0, 1.0, 0.8)), ("duration_1.25", (1.0, 1.0, 1.25)),
                    ("pitch_0.8", (0.8, 1.0, 1.0)), ("pitch_1.2", (1.2, 1.0, 1.0)),
                    ("energy_0.8", (1.0, 0.8, 1.0)), ("energy_1.2", (1.0, 1.2, 1.0))]:
        r = synthesize(model, vocoder, configs, spk_id, seq, controls=c)
        if r["wav"] is None:
            ctrl.append({"condition": name, "error": "zero-length prediction"})
            continue
        wavfile.write(os.path.join(out, "controls", "{}.wav".format(name)), sr, r["wav"])
        x = r["wav"].astype(np.float64) / 32768.0
        ctrl.append({"condition": name, "pitch_control": c[0], "energy_control": c[1], "duration_control": c[2],
                     "audio_s": len(r["wav"]) / sr, "median_f0_hz": f0_stats(r["wav"], sr, frame_ms),
                     "rms": float(np.sqrt(np.mean(x * x)))})
    report["controls"] = {"text": controls_text, "conditions": ctrl,
                          "note": "pitch/energy controls scale the normalised predictions in the reference code"}

    # ---------------------------------------------------------------- 5. CPU speed
    try:
        if not n:
            raise RuntimeError("skipped (no validation utterances in this mode)")
        cpu = torch.device("cpu")
        m_cpu = copy.deepcopy(model).to(cpu)
        v_cpu = get_vocoder(model_cfg, cpu)
        torch.set_num_threads(os.cpu_count() or 1)
        items = [ds[i] for i in range(min(args.cpu_utts, n))]
        synthesize(m_cpu, v_cpu, configs, items[0]["speaker"], items[0]["text"], dev=cpu)  # warm-up
        a = v = au = 0.0
        for s in items:
            r = synthesize(m_cpu, v_cpu, configs, s["speaker"], s["text"], dev=cpu)
            a, v, au = a + r["acoustic_s"], v + r["vocoder_s"], au + (len(r["wav"]) / sr if r["wav"] is not None else 0)
        report["speed_cpu"] = {"utterances": len(items), "audio_s": au, "rtf_acoustic": a / au, "rtf_vocoder": v / au,
                               "rtf_total": (a + v) / au, "threads": torch.get_num_threads(),
                               "note": "batch 1, warm, float32"}
        del m_cpu, v_cpu
    except Exception as e:  # noqa: BLE001
        report["speed_cpu"] = {"error": "{}: {}".format(type(e).__name__, e)}
    if "speed_gpu" in report:
        g = report["speed_gpu"]
        g["rtf_total"] = g["rtf_acoustic"] + g["rtf_vocoder"]
    json.dump(report, open(os.path.join(out, "metrics.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    # ---------------------------------------------------------------- 2. intelligibility (ASR)
    del model, vocoder
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    try:
        asr = load_asr(args.asr_model)
        wnorm = whisper_english_normaliser(asr) if lang == "en" else None
        hyp_s = transcribe(asr, [a["synth"] for a in asr_items], sr, lang)
        hyp_c = transcribe(asr, [a["copy"] for a in asr_items], sr, lang)
        asr_rows = []
        for a, hs, hc in zip(asr_items, hyp_s, hyp_c):
            es_, ec = M.error_rates(a["ref"], hs), M.error_rates(a["ref"], hc)
            asr_rows.append({"id": a["id"], "speaker": a["speaker"], "reference": a["ref"], "synth_hyp": hs,
                             "copy_hyp": hc, "synth": es_, "copy": ec})
            if wnorm:
                asr_rows[-1]["synth_wn"] = M.error_rates(a["ref"], hs, wnorm)
                asr_rows[-1]["copy_wn"] = M.error_rates(a["ref"], hc, wnorm)
        with open(os.path.join(out, "asr_validation.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "speaker", "reference", "synth_transcript", "synth_cer", "copy_synthesis_transcript",
                        "copy_cer"])
            for r in asr_rows:
                w.writerow([r["id"], r["speaker"], r["reference"], r["synth_hyp"], round(r["synth"]["cer"], 4),
                            r["copy_hyp"], round(r["copy"]["cer"], 4)])
        report["intelligibility"] = {} if not asr_rows else {
            "asr_model": args.asr_model, "utterances": len(asr_rows),
            "synthesis": M.corpus_rates([r["synth"] for r in asr_rows]),
            "copy_synthesis_ceiling": M.corpus_rates([r["copy"] for r in asr_rows]),
            "reference_text": "the validation transcript (.lab text used for alignment)"}
        if wnorm and asr_rows:
            report["intelligibility"]["whisper_english_normaliser"] = {
                "synthesis": M.corpus_rates([r["synth_wn"] for r in asr_rows]),
                "copy_synthesis_ceiling": M.corpus_rates([r["copy_wn"] for r in asr_rows]),
                "note": "Whisper's EnglishTextNormalizer applied to reference and transcript (numbers, spellings)"}
        sent_ok = [r for r in sent_rows if r.get("wav") is not None]
        for r, h in zip(sent_ok, transcribe(asr, [r["wav"] for r in sent_ok], sr, lang)):
            r["asr_transcript"] = h
            r["asr_vs_model_input"] = M.error_rates(r["model_input"], h)
            if wnorm:  # against the ORIGINAL text ("10:45 AM"), both sides through Whisper's normaliser
                r["asr_vs_original_whisper_norm"] = M.error_rates(r["text"], h, wnorm)
    except Exception as e:  # noqa: BLE001  (ASR is an extra; its failure must not lose the other metrics)
        report["intelligibility"] = {"error": "{}: {}".format(type(e).__name__, e)}
    for r in sent_rows:
        r.pop("wav", None)
    report["sentences"] = sent_rows
    json.dump(report, open(os.path.join(out, "metrics.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in ("validation", "intelligibility", "speed_gpu", "speed_cpu")
                      if k in report}, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--speaker", required=True, help="test voice for the test/stress/control sentences")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_utts", type=int, default=0, help="validation utterances (0 = all)")
    ap.add_argument("--asr_utts", type=int, default=150)
    ap.add_argument("--cpu_utts", type=int, default=20)
    ap.add_argument("--save_samples", type=int, default=6)
    ap.add_argument("--asr_model", default="openai/whisper-large-v3")
    ap.add_argument("--sentences_only", action="store_true",
                    help="skip the validation set and CPU timing: test/stress sentences, controls and their ASR")
    main(ap.parse_args())
