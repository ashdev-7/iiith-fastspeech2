"""Speaker adaptation: fine-tune the trained multi-speaker model on the TRAINING clips of one voice.

Why: on English the decoder averages over 110 speakers, so the test voice comes out over-smoothed and
breathier than its own recordings (report Section 6.3: HNR natural 16.3 dB, copy-synthesis 16.7, oracle 15.1).
Fine-tuning only the decoder side (decoder, mel_linear, postnet, speaker embedding; optionally the
variance adaptor) on that voice's training clips is standard speaker adaptation (AdaSpeech, Chen et al.
2021). The encoder stays frozen. Data: the voice's clips in train.txt only; its clips in val.txt (never
trained on) choose the step (early stopping) and are reported.

Outputs in --out_dir: adapted.pth.tar (best step by free-running MCD on the voice's validation clips),
adapt_log.csv (train loss), adapt_val.csv (per evaluation: free MCD-DTW, oracle MCD, GV ratio, pitch r),
adapt_summary.json.
"""
import argparse
import csv
import json
import os
import random

import numpy as np
import torch

import eval_metrics as M
from dataset import Dataset
from evaluate_val import oracle_mel, synthesize
from inference import device, load_model
from model import FastSpeech2Loss
from utils.config import load_configs
from utils.tools import to_device

PARTS = {
    "decoder": ("decoder", "mel_linear", "postnet", "speaker_emb"),
    "decoder_var": ("decoder", "mel_linear", "postnet", "speaker_emb", "variance_adaptor"),
}


def trainable(model, parts):
    names = PARTS[parts]
    params = []
    for name, p in model.named_parameters():
        p.requires_grad = name.split(".")[0] in names
        if p.requires_grad:
            params.append(p)
    return params


def gv_ratio(pred, gt):
    return float(np.mean(np.var(pred, axis=0) / np.maximum(np.var(gt, axis=0), 1e-8)))


def validate(model, configs, samples):
    model.eval()
    rows = []
    for s in samples:
        r = synthesize(model, None, configs, s["speaker"], s["text"])
        if not r["mel_len"]:
            continue
        g = s["mel"]
        cg = M.mel_cepstra(g)
        o = oracle_mel(model, s)
        nz = s["duration"] > 0
        pr = np.corrcoef(r["p"][nz], s["pitch"][nz])[0, 1] if nz.sum() > 2 else float("nan")
        rows.append({"mcd_free": M.mcd_dtw(M.mel_cepstra(r["mel"]), cg)[0],
                     "mcd_oracle": M.mcd_dtw(M.mel_cepstra(o), cg)[0],
                     "gv_free": gv_ratio(r["mel"], g), "gv_oracle": gv_ratio(o, g), "pitch_r": float(pr)})
    model.train()
    keys = ("mcd_free", "mcd_oracle", "gv_free", "gv_oracle", "pitch_r")
    if not rows:  # every clip predicted zero frames
        return {k: float("inf") if k.startswith("mcd") else float("nan") for k in keys}
    return {k: float(np.nanmean([x[k] for x in rows])) for k in keys}


def main(a):
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    configs = load_configs(a.config, a.data_root)
    pre_cfg, model_cfg, train_cfg = configs
    model, base_step = load_model(configs, a.checkpoint)
    os.makedirs(a.out_dir, exist_ok=True)

    tr = Dataset("train.txt", pre_cfg, train_cfg, sort=False, drop_last=False)
    va = Dataset("val.txt", pre_cfg, train_cfg, sort=False, drop_last=False)
    train = [tr[i] for i in range(len(tr)) if tr.speaker[i] == a.speaker]
    val = [va[i] for i in range(len(va)) if va.speaker[i] == a.speaker]
    assert train and val, "speaker has no train or no val clips"
    assert not ({s["id"] for s in train} & {s["id"] for s in val}), "train/val overlap"
    print("adapting on {} train clips, validating on {}".format(len(train), len(val)), flush=True)

    params = trainable(model, a.parts)
    opt = torch.optim.Adam(params, lr=a.lr, betas=(0.9, 0.98), eps=1e-9)
    Loss = FastSpeech2Loss(pre_cfg, model_cfg).to(device)
    model.train()

    vlog = open(os.path.join(a.out_dir, "adapt_val.csv"), "w", newline="")
    vw = csv.writer(vlog)
    vw.writerow(["step", "mcd_free", "mcd_oracle", "gv_free", "gv_oracle", "pitch_r"])
    tlog = open(os.path.join(a.out_dir, "adapt_log.csv"), "w", newline="")
    tw = csv.writer(tlog)
    tw.writerow(["step", "total", "mel", "postnet", "pitch", "energy", "duration"])

    def evaluate(step):
        v = validate(model, configs, val)
        vw.writerow([step] + [round(v[k], 4) for k in ("mcd_free", "mcd_oracle", "gv_free", "gv_oracle", "pitch_r")])
        vlog.flush()
        print("step {} val {}".format(step, {k: round(x, 3) for k, x in v.items()}), flush=True)
        return v

    best = evaluate(0)
    best["step"] = 0
    history = [dict(best)]
    torch.save({"model": model.state_dict(), "step": base_step}, os.path.join(a.out_dir, "adapted.pth.tar"))
    bad = 0
    for step in range(1, a.steps + 1):
        idx = random.sample(range(len(train)), min(a.batch_size, len(train)))
        batch = to_device(tr.reprocess(train, idx), device)
        out = model(*(batch[2:]))
        losses = Loss(batch, out)
        opt.zero_grad()
        losses[0].backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 10 == 0:
            tw.writerow([step] + [round(float(x), 5) for x in losses])
            tlog.flush()
        if step % a.eval_every == 0:
            v = evaluate(step)
            v["step"] = step
            history.append(dict(v))
            if v["mcd_free"] < best["mcd_free"] - 1e-3:
                best, bad = v, 0
                torch.save({"model": model.state_dict(), "step": base_step,
                            "adapted": {"speaker": a.speaker, "parts": a.parts, "steps": step, "lr": a.lr}},
                           os.path.join(a.out_dir, "adapted.pth.tar"))
            else:
                bad += 1
                if bad >= a.patience:
                    print("early stop at step", step, flush=True)
                    break
    json.dump({"speaker": a.speaker, "parts": a.parts, "lr": a.lr, "batch_size": a.batch_size,
               "train_clips": len(train), "val_clips": len(val), "base_step": base_step,
               "start": history[0], "best": best, "history": history},
              open(os.path.join(a.out_dir, "adapt_summary.json"), "w"), indent=1)
    print("best", best, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--parts", choices=sorted(PARTS), default="decoder")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    main(ap.parse_args())
