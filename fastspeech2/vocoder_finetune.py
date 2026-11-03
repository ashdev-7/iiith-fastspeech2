"""Fine-tune the HiFi-GAN universal vocoder on ground-truth-aligned (GTA) mels of our FastSpeech 2.

Why: the FastSpeech 2 decoder over-smooths (English free-running mels keep 73-89 % of the natural per-bin
variance over time; copy-synthesis keeps 95-101 %), and the universal vocoder was trained on real mels only.
Fine-tuning the vocoder on the acoustic model's own outputs, paired with the real audio, teaches it to
produce sharp audio from smoothed mels (the HiFi-GAN fine-tuning recipe; NVIDIA's FastPitch+HiFi-GAN
recipe does the same). Only the assignment's training data is used; validation clips are only scored.

GTA mel = FastSpeech 2 with ground-truth duration, pitch and energy (teacher forcing), so its frames line
up with the recording. The recording segment is cut exactly as the reference preprocessor cuts it
(TextGrid start/end trim), and checked: the mel recomputed from that segment must equal the stored
ground-truth mel.

Generator: initialised from generator_universal (weights with weight norm). Discriminators (MPD + MSD) are
not distributed with the reference, so they start from scratch: --d_warmup steps update only them,
then generator and discriminators train together (loss: adversarial + 2 x feature matching + 45 x mel L1,
as in HiFi-GAN).

Outputs in --out_dir: generator_ft.pth.tar ({"generator": state_dict, "step"}), generator_ft_best.pth.tar
(lowest validation mel L1), losses.csv, val.csv, gta_check.json.
"""
import argparse
import csv
import json
import os
import random
import time
import types

import numpy as np
import soundfile as sf
import tgt
import torch
import torch.nn.functional as F

import hifigan
from audio.stft import TacotronSTFT
from audio.tools import get_mel_from_wav
from dataset import Dataset
from evaluate_val import oracle_mel
from hifigan.discriminators import (MultiPeriodDiscriminator, MultiScaleDiscriminator, discriminator_loss,
                                    feature_loss, generator_loss)
from inference import device, load_model
from preprocessor.preprocessor import Preprocessor
from utils.config import load_configs
from utils.model import HIFIGAN_DIR, _hifigan_ckpt


class DiffMel(torch.nn.Module):
    """Differentiable log-mel identical in parameters to the reference TacotronSTFT
    (whose magnitude is detached and cannot carry gradients)."""

    def __init__(self, stft_module):
        super().__init__()
        self.n_fft = stft_module.stft_fn.filter_length
        self.hop = stft_module.stft_fn.hop_length
        self.win = stft_module.stft_fn.win_length
        self.register_buffer("mel_basis", stft_module.mel_basis.clone().float())
        self.register_buffer("window", torch.hann_window(self.win))

    def forward(self, y):  # y: (B, T) in [-1, 1]
        spec = torch.stft(y, self.n_fft, self.hop, self.win, self.window, center=True, pad_mode="reflect",
                          return_complex=True)
        mag = torch.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-9)
        return torch.log(torch.clamp(torch.matmul(self.mel_basis, mag), min=1e-5))


def build_gta(model, configs, split, stft, limit=None):
    pre_cfg, _, train_cfg = configs
    sr = pre_cfg["preprocessing"]["audio"]["sampling_rate"]
    hop = pre_cfg["preprocessing"]["stft"]["hop_length"]
    raw, pre = pre_cfg["path"]["raw_path"], pre_cfg["path"]["preprocessed_path"]
    ns = types.SimpleNamespace(sampling_rate=sr, hop_length=hop)
    ds = Dataset(split, pre_cfg, train_cfg, sort=False, drop_last=False)
    items, diffs = [], []
    n = len(ds) if limit is None else min(limit, len(ds))
    for i in range(n):
        s = ds[i]
        spk = ds.speaker[i]
        tier = tgt.io.read_textgrid(os.path.join(pre, "TextGrid", spk, s["id"] + ".TextGrid"),
                                    include_empty_intervals=True).get_tier_by_name("phones")
        _, _, start, end = Preprocessor.get_alignment(ns, tier)
        wav, wsr = sf.read(os.path.join(raw, spk, s["id"] + ".wav"), dtype="float32")
        assert wsr == sr, (wsr, sr)
        wav = wav[int(sr * start): int(sr * end)]
        T = s["mel"].shape[0]
        if i < 20:  # the cut must reproduce the stored mel exactly
            m, _ = get_mel_from_wav(wav, stft)
            diffs.append(float(np.abs(m[:, :T].T - s["mel"]).max()))
        need = T * hop
        wav = np.pad(wav, (0, max(0, need - len(wav))))[:need]
        items.append({"id": s["id"], "speaker": spk, "mel": oracle_mel(model, s).astype(np.float32),
                      "wav": np.clip(wav, -1, 1).astype(np.float32)})
        if i % 500 == 0:
            print("gta {} {}/{}".format(split, i, n), flush=True)
    return items, diffs


def batch(items, bsz, frames, hop):
    mels, wavs = [], []
    for it in random.choices(items, k=bsz):
        T = it["mel"].shape[0]
        if T <= frames:
            m = np.pad(it["mel"], ((0, frames - T), (0, 0)), constant_values=float(it["mel"].min()))
            w = np.pad(it["wav"], (0, frames * hop - len(it["wav"])))
        else:
            f = random.randint(0, T - frames)
            m, w = it["mel"][f:f + frames], it["wav"][f * hop:(f + frames) * hop]
        mels.append(m.T)
        wavs.append(w)
    return (torch.from_numpy(np.stack(mels)).to(device), torch.from_numpy(np.stack(wavs)).to(device))


def validate(G, items, mel_fn):
    G.eval()
    l1, gv = [], []
    with torch.no_grad():
        for it in items:
            y_hat = G(torch.from_numpy(it["mel"].T).unsqueeze(0).to(device)).squeeze(1)
            m_hat = mel_fn(y_hat.clamp(-1, 1))
            m_ref = mel_fn(torch.from_numpy(it["wav"]).unsqueeze(0).to(device))
            n = min(m_hat.shape[-1], m_ref.shape[-1])
            l1.append(float(F.l1_loss(m_hat[..., :n], m_ref[..., :n])))
            gv.append(float((m_hat[..., :n].var(-1) / m_ref[..., :n].var(-1).clamp(min=1e-8)).mean()))
    G.train()
    return float(np.mean(l1)), float(np.mean(gv))


def save(G, step, path):
    tmp = path + ".tmp"
    torch.save({"generator": G.state_dict(), "step": step}, tmp)
    os.replace(tmp, path)


def main(a):
    torch.manual_seed(1234)
    random.seed(1234)
    torch.backends.cudnn.benchmark = True
    configs = load_configs(a.config, a.data_root)
    pre_cfg = configs[0]
    hop = pre_cfg["preprocessing"]["stft"]["hop_length"]
    os.makedirs(a.out_dir, exist_ok=True)
    st = pre_cfg["preprocessing"]["stft"]
    me = pre_cfg["preprocessing"]["mel"]
    stft = TacotronSTFT(st["filter_length"], st["hop_length"], st["win_length"], me["n_mel_channels"],
                        pre_cfg["preprocessing"]["audio"]["sampling_rate"], me["mel_fmin"], me["mel_fmax"])

    model, step = load_model(configs, a.checkpoint)
    t0 = time.time()
    train, d_tr = build_gta(model, configs, "train.txt", stft, a.limit)
    val, d_va = build_gta(model, configs, "val.txt", stft, a.val_utts)
    del model
    torch.cuda.empty_cache()
    check = {"fs2_step": step, "train_utterances": len(train), "val_utterances": len(val),
             "max_abs_mel_diff_recomputed_vs_stored": max(d_tr + d_va), "gta_minutes": round((time.time() - t0) / 60, 1)}
    json.dump(check, open(os.path.join(a.out_dir, "gta_check.json"), "w"), indent=1)
    print(check, flush=True)
    if check["max_abs_mel_diff_recomputed_vs_stored"] > 1e-3:
        raise SystemExit("recording cut does not reproduce the stored mel: GTA pairs would be misaligned")

    h = hifigan.AttrDict(json.load(open(os.path.join(HIFIGAN_DIR, "config.json"))))
    G = hifigan.Generator(h).to(device)
    G.load_state_dict(torch.load(_hifigan_ckpt("generator_universal.pth.tar"), map_location=device,
                                 weights_only=True)["generator"])
    mpd, msd = MultiPeriodDiscriminator().to(device), MultiScaleDiscriminator().to(device)
    mel_fn = DiffMel(stft).to(device)
    opt_g = torch.optim.AdamW(G.parameters(), a.lr_g, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(list(mpd.parameters()) + list(msd.parameters()), a.lr_d, betas=(0.8, 0.99))

    frames = a.segment // hop
    l1_0, gv_0 = validate(G, val, mel_fn)
    best = l1_0
    vrows = [{"step": 0, "val_mel_l1": round(l1_0, 5), "val_gv_ratio": round(gv_0, 4)}]
    print("step 0 (universal): val mel L1 {:.4f}, GV ratio {:.3f}".format(l1_0, gv_0), flush=True)
    lcsv = open(os.path.join(a.out_dir, "losses.csv"), "w", newline="")
    lw = csv.writer(lcsv)
    lw.writerow(["step", "loss_d", "loss_adv", "loss_fm", "mel_l1", "steps_per_s"])
    acc, t_start, t_log = [], time.time(), time.time()
    s = 0
    while s < a.max_steps and (time.time() - t_start) / 60 < a.max_minutes:
        s += 1
        mel, y = batch(train, a.batch, frames, hop)
        y = y.unsqueeze(1)
        warm = s <= a.d_warmup
        if warm:
            with torch.no_grad():
                y_g = G(mel)
        else:
            y_g = G(mel)
        # discriminators
        opt_d.zero_grad(set_to_none=True)
        r, g, _, _ = mpd(y, y_g.detach())
        loss_d = discriminator_loss(r, g)
        r, g, _, _ = msd(y, y_g.detach())
        loss_d = loss_d + discriminator_loss(r, g)
        loss_d.backward()
        opt_d.step()
        # generator
        mel_l1 = F.l1_loss(mel_fn(y_g.squeeze(1).clamp(-1, 1)), mel_fn(y.squeeze(1)))
        if warm:
            adv = fm = torch.zeros(())
        else:
            opt_g.zero_grad(set_to_none=True)
            _, g_p, f_r_p, f_g_p = mpd(y, y_g)
            _, g_s, f_r_s, f_g_s = msd(y, y_g)
            fm = feature_loss(f_r_p, f_g_p) + feature_loss(f_r_s, f_g_s)
            adv = generator_loss(g_p) + generator_loss(g_s)
            loss_g = adv + fm + 45 * mel_l1
            loss_g.backward()
            opt_g.step()
        vals = [float(loss_d), float(adv), float(fm), float(mel_l1)]
        if not all(np.isfinite(vals)):
            raise SystemExit("non-finite loss at step {}: {}".format(s, vals))
        acc.append(vals)
        if s % 100 == 0:
            m = np.mean(acc, 0)
            lw.writerow([s] + [round(float(x), 5) for x in m] + [round(100 / (time.time() - t_log), 3)])
            lcsv.flush()
            acc, t_log = [], time.time()
        if s % a.val_every == 0:
            l1, gv = validate(G, val, mel_fn)
            vrows.append({"step": s, "val_mel_l1": round(l1, 5), "val_gv_ratio": round(gv, 4)})
            print("step {}: val mel L1 {:.4f}, GV ratio {:.3f}".format(s, l1, gv), flush=True)
            if s > a.d_warmup and l1 < best:
                best = l1
                save(G, s, os.path.join(a.out_dir, "generator_ft_best.pth.tar"))
            with open(os.path.join(a.out_dir, "val.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(vrows[0].keys()))
                w.writeheader()
                w.writerows(vrows)
        if s % a.save_every == 0:
            save(G, s, os.path.join(a.out_dir, "generator_ft.pth.tar"))
    save(G, s, os.path.join(a.out_dir, "generator_ft.pth.tar"))
    l1, gv = validate(G, val, mel_fn)
    vrows.append({"step": s, "val_mel_l1": round(l1, 5), "val_gv_ratio": round(gv, 4)})
    with open(os.path.join(a.out_dir, "val.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(vrows[0].keys()))
        w.writeheader()
        w.writerows(vrows)
    print("done: {} steps, val mel L1 {:.4f} (universal {:.4f}), GV ratio {:.3f} (universal {:.3f})".format(
        s, l1, l1_0, gv, gv_0), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--checkpoint", required=True, help="FastSpeech 2 checkpoint (for the GTA mels)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_minutes", type=float, default=240)
    ap.add_argument("--max_steps", type=int, default=10 ** 9)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--segment", type=int, default=8192)
    ap.add_argument("--lr_g", type=float, default=1e-4)
    ap.add_argument("--lr_d", type=float, default=2e-4)
    ap.add_argument("--d_warmup", type=int, default=1000)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--val_utts", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None, help="train utterances (smoke tests)")
    main(ap.parse_args())
