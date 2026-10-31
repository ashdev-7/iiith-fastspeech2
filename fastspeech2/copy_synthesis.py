"""HiFi-GAN copy-synthesis check (plan.txt G1): ground-truth mel -> vocoder -> wav, per speaker.

The vocoder bounds the quality any FastSpeech 2 model can reach, so this measures, per speaker, how
well the universal HiFi-GAN reproduces that speaker. Metric: mean L1 distance between the
ground-truth log-mel and the log-mel re-extracted from the vocoded wav (lower is better), on up to
--per_speaker validation utterances. The wavs are written for listening.

  python copy_synthesis.py --config ../Telugu/config.yaml --data_root D --out_dir D/copy_synthesis
"""
import argparse
import collections
import json
import os

import numpy as np
import torch
from scipy.io import wavfile

import audio as Audio
from utils.config import load_configs
from utils.model import get_vocoder, vocoder_infer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    preprocess_config, model_config, _ = load_configs(args.config, args.data_root)
    pre = preprocess_config["path"]["preprocessed_path"]
    p = preprocess_config["preprocessing"]
    stft = Audio.stft.TacotronSTFT(p["stft"]["filter_length"], p["stft"]["hop_length"], p["stft"]["win_length"],
                                   p["mel"]["n_mel_channels"], p["audio"]["sampling_rate"],
                                   p["mel"]["mel_fmin"], p["mel"]["mel_fmax"])
    vocoder = get_vocoder(model_config, device)
    os.makedirs(args.out_dir, exist_ok=True)

    by_spk = collections.defaultdict(list)
    with open(os.path.join(pre, "val.txt"), encoding="utf-8") as f:
        for line in f:
            basename, speaker = line.split("|")[:2]
            by_spk[speaker].append(basename)

    report = {}
    for speaker in sorted(by_spk):
        dists = []
        for basename in sorted(by_spk[speaker])[: args.per_speaker]:
            mel = np.load(os.path.join(pre, "mel", "{}-mel-{}.npy".format(speaker, basename)))  # (T, 80)
            mel_t = torch.from_numpy(mel.T).float().unsqueeze(0).to(device)
            wav = vocoder_infer(mel_t, vocoder, model_config, preprocess_config)[0]
            if args.save_wavs:
                wavfile.write(os.path.join(args.out_dir, "{}_{}.wav".format(speaker, basename)),
                              p["audio"]["sampling_rate"], wav)
            re_mel, _ = Audio.tools.get_mel_from_wav(wav.astype(np.float32) / p["audio"]["max_wav_value"], stft)
            n = min(re_mel.shape[1], mel.shape[0])
            dists.append(float(np.mean(np.abs(re_mel[:, :n].T - mel[:n]))))
        report[speaker] = {"utterances": len(dists), "mel_l1_mean": float(np.mean(dists)) if dists else None,
                           "mel_l1_std": float(np.std(dists)) if dists else None}
    ranked = sorted((s for s in report if report[s]["mel_l1_mean"] is not None), key=lambda s: report[s]["mel_l1_mean"])
    out = {"metric": "mean |log-mel(gt) - log-mel(HiFi-GAN(gt))|, lower is better", "per_speaker": report,
           "ranked": ranked}
    with open(os.path.join(args.out_dir, "copy_synthesis.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out)[:2000])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--per_speaker", type=int, default=20)
    parser.add_argument("--save_wavs", action="store_true")
    main(parser.parse_args())
