"""Synthesise the assignment's test sentences and export the variance-adaptor predictions (plan.txt 7).

For every test item of the language in tests/test_sentences.json:
  original text (kept verbatim) -> frontend -> phones -> FastSpeech 2 (no targets) ->
  durations d = clamp(round(exp(log_d) - 1), 0)   [reference conversion]
  pitch Hz  = pred * std + mean   (stats.json; the reference normalises linear Hz)
  energy    = pred * std + mean   (frame L2 STFT magnitude, arbitrary units)
  postnet mel -> HiFi-GAN universal -> 22050 Hz 16-bit wav
Writes <out_dir>/test_generated_audios/<ID>_<lang>.wav, variance_raw/<ID>.json (+ .npz),
variance_adaptor_plots/{duration,pitch,energy}/<ID>_<kind>.png, frontend_trace.json and
inference_report.json (with per-item timing: acoustic model and vocoder seconds, real-time factor).

  python inference.py --config ../Telugu/config.yaml --data_root D --checkpoint D/output/te/ckpt/60000.pth.tar \
      --speaker te_female --out_dir ../Results/Telugu
"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch
from scipy.io import wavfile

from model import FastSpeech2
from plotting import TITLE_RENDERING, find_font, plot_variances
from text import text_to_sequence
from text.symbols import get_symbols
from utils.config import load_configs
from utils.model import get_vocoder, vocoder_infer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LEXICON = os.path.join(HERE, "lexicon", "librispeech-lexicon.txt")


def frontend(text, lang):
    if lang == "en":
        from text.english_frontend import text_to_phones
        return text_to_phones(text, LEXICON)
    from text.telugu.g2p import sentence_to_phones
    from text.telugu.normalize import normalize
    phones, trace = sentence_to_phones(text)
    return phones, {"original": text, "model_input": normalize(text),
                    "words": [{"token": t, "phones": p} for t, p in trace]}


def symbol_table_hash(lang):
    return hashlib.sha256("\n".join(get_symbols(lang)).encode("utf-8")).hexdigest()


def load_model(configs, checkpoint):
    preprocess_config, model_config, _ = configs
    model = FastSpeech2(preprocess_config, model_config).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("step")


def _now(dev):
    """Wall clock after all queued GPU work has finished (CUDA calls are asynchronous)."""
    import time
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def generated_f0(wav, sr, period_ms):
    import pyworld as pw
    x = wav.astype(np.float64) / 32768.0
    f0, t = pw.dio(x, sr, frame_period=period_ms)
    return pw.stonemask(x, f0, t, sr)


def main(args):
    configs = load_configs(args.config, args.data_root)
    preprocess_config, model_config, _ = configs
    lang = preprocess_config["preprocessing"]["text"]["language"]
    pre = preprocess_config["path"]["preprocessed_path"]
    p = preprocess_config["preprocessing"]
    sr, hop = p["audio"]["sampling_rate"], p["stft"]["hop_length"]
    frame_ms = 1000.0 * hop / sr
    with open(os.path.join(pre, "stats.json"), encoding="utf-8") as f:
        stats = json.load(f)
    with open(os.path.join(pre, "speakers.json"), encoding="utf-8") as f:
        speakers = json.load(f)
    if args.speaker not in speakers:
        raise SystemExit("speaker {!r} not in speakers.json".format(args.speaker))
    pitch_mean, pitch_std = stats["pitch"][2:4]
    energy_mean, energy_std = stats["energy"][2:4]

    model, step = load_model(configs, args.checkpoint)
    vocoder = get_vocoder(model_config, device)
    telugu_font = find_font(extra_dirs=[args.font_dir] if args.font_dir else ())

    with open(args.sentences, encoding="utf-8") as f:
        items = [it for it in json.load(f)["items"] if it["lang"] == lang]
    out = args.out_dir
    for d in ("test_generated_audios", "variance_raw"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    report = {"checkpoint": os.path.basename(args.checkpoint), "step": step, "speaker": args.speaker,
              "speaker_id": speakers[args.speaker], "symbol_table_sha256": symbol_table_hash(lang),
              "telugu_font": telugu_font, "controls": {"pitch": args.pitch_control, "energy": args.energy_control,
                                                       "duration": args.duration_control},
              "items": {}}
    traces = {}
    warmed_up = False
    report["device"] = str(device) + (" ({})".format(torch.cuda.get_device_name(0)) if device.type == "cuda" else "")
    for it in items:
        item_id, text = it["id"], it["canonical"]
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != it["canonical_sha256"]:
            raise SystemExit("{}: test text does not match its recorded sha256".format(item_id))
        phones, trace = frontend(text, lang)
        seq = text_to_sequence("{" + " ".join(phones) + "}", [], lang)
        trace["phones"], trace["ids"] = phones, seq
        traces[item_id] = trace

        texts = torch.from_numpy(np.array([seq])).long().to(device)
        src_lens = torch.tensor([len(seq)], device=device)
        spk = torch.tensor([speakers[args.speaker]], device=device).long()
        d_control = args.duration_control
        scales = trace.get("duration_scale")
        if scales and any(x != 1.0 for x in scales):  # per-phone duration control (letter names)
            assert len(scales) == len(seq), (len(scales), len(seq))
            d_control = torch.tensor([scales], device=device, dtype=torch.float32) * args.duration_control

        def acoustic():
            with torch.no_grad():
                return model(spk, texts, src_lens, len(seq), p_control=args.pitch_control,
                             e_control=args.energy_control, d_control=d_control)

        if not warmed_up:  # first call pays CUDA/cuDNN initialisation; keep it out of the timings
            acoustic()
            warmed_up = True
        t0 = _now(device)
        output = acoustic()
        t_acoustic = _now(device) - t0
        postnet_mel, p_pred, e_pred, log_d, d_rounded, mel_lens = (
            output[1], output[2], output[3], output[4], output[5], output[9])
        durs = d_rounded[0, : len(seq)].cpu().numpy().astype(int)
        mel_len = int(mel_lens[0].item())
        if mel_len == 0:
            raise SystemExit("{}: the model predicted zero frames for every phone".format(item_id))
        mel = postnet_mel[0, :mel_len].transpose(0, 1).unsqueeze(0)
        t0 = _now(device)
        wav = vocoder_infer(mel, vocoder, model_config, preprocess_config)[0]
        t_vocoder = _now(device) - t0
        audio_s = len(wav) / sr
        timing = {"acoustic_s": round(t_acoustic, 5), "vocoder_s": round(t_vocoder, 5), "audio_s": round(audio_s, 3),
                  "rtf_acoustic": round(t_acoustic / audio_s, 5),
                  "rtf_total": round((t_acoustic + t_vocoder) / audio_s, 5)}
        wav_path = os.path.join(out, "test_generated_audios", "{}_{}.wav".format(item_id, lang))
        wavfile.write(wav_path, sr, wav)

        pitch_hz = p_pred[0, : len(seq)].cpu().numpy() * pitch_std + pitch_mean
        energy = e_pred[0, : len(seq)].cpu().numpy() * energy_std + energy_mean
        f0_period = frame_ms
        f0 = generated_f0(wav, sr, f0_period)
        rms = float(np.sqrt(np.mean((wav.astype(np.float64) / 32768.0) ** 2)))
        checks = {"sum_d_equals_mel_len": int(durs.sum()) == mel_len, "len_d_equals_phones": len(durs) == len(phones),
                  "wav_samples": int(len(wav)), "mel_frames_x_hop": mel_len * hop,
                  "wav_finite": bool(np.all(np.isfinite(wav))), "wav_rms": rms, "wav_not_silent": rms > 1e-3,
                  "zero_duration_phones": int((durs == 0).sum())}
        rec = {"id": item_id, "lang": lang, "original_text": text, "model_input": trace["model_input"],
               "phones": phones, "phone_ids": seq, "duration_frames": durs.tolist(),
               "duration_ms": (durs * frame_ms).round(2).tolist(), "log_duration_pred": log_d[0, : len(seq)].cpu().tolist(),
               "pitch_hz": pitch_hz.tolist(), "pitch_normalised": p_pred[0, : len(seq)].cpu().tolist(),
               "energy": energy.tolist(), "energy_normalised": e_pred[0, : len(seq)].cpu().tolist(),
               "energy_unit": "frame L2 norm of the STFT magnitude (arbitrary units)",
               "frame_ms": frame_ms, "generated_f0_hz": f0.tolist(), "generated_f0_period_ms": f0_period,
               "speaker": args.speaker, "checkpoint_step": step, "checks": checks}
        with open(os.path.join(out, "variance_raw", "{}.json".format(item_id)), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        np.savez(os.path.join(out, "variance_raw", "{}.npz".format(item_id)), duration_frames=durs, pitch_hz=pitch_hz,
                 energy=energy, generated_f0_hz=f0, mel=mel[0].cpu().numpy())
        plots = plot_variances(rec, os.path.join(out, "variance_adaptor_plots"), telugu_font)
        report["items"][item_id] = {"wav": os.path.relpath(wav_path, out), "seconds": round(len(wav) / sr, 3),
                                    "phones": len(phones), "checks": checks, "timing": timing,
                                    "plots": {k: os.path.relpath(v, out) for k, v in plots.items()}}
        print(item_id, json.dumps(checks), flush=True)

    tm = [r["timing"] for r in report["items"].values()]
    if tm:
        audio = sum(t["audio_s"] for t in tm)
        report["timing_summary"] = {"items": len(tm), "audio_s": round(audio, 3),
                                    "rtf_acoustic": round(sum(t["acoustic_s"] for t in tm) / audio, 5),
                                    "rtf_total": round(sum(t["acoustic_s"] + t["vocoder_s"] for t in tm) / audio, 5),
                                    "note": "batch 1, warm (one untimed warm-up call), excludes text frontend, "
                                            "wav writing and plotting"}
    report["title_rendering"] = TITLE_RENDERING["mode"] or "Latin text only"
    with open(os.path.join(out, "frontend_trace.json"), "w", encoding="utf-8") as f:
        json.dump(traces, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out, "inference_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    failed = [i for i, r in report["items"].items()
              if not (r["checks"]["sum_d_equals_mel_len"] and r["checks"]["len_d_equals_phones"]
                      and r["checks"]["wav_finite"] and r["checks"]["wav_not_silent"])]
    if failed:
        raise SystemExit("consistency checks failed for {}".format(failed))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_root", default=None, help="root holding preprocessed/<lang>/stats.json, speakers.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--speaker", required=True, help="speaker name from speakers.json")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sentences", default=os.path.join(REPO, "tests", "test_sentences.json"))
    parser.add_argument("--font_dir", default=None, help="extra directory with a Telugu .ttf (e.g. Noto Sans Telugu)")
    parser.add_argument("--pitch_control", type=float, default=1.0)
    parser.add_argument("--energy_control", type=float, default=1.0)
    parser.add_argument("--duration_control", type=float, default=1.0)
    main(parser.parse_args())
