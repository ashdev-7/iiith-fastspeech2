"""FastSpeech 2 training (adapted from ming024/FastSpeech2 train.py).

Changes from the reference (see MODIFICATIONS.md):
  * one config.yaml per language (utils/config.py) and a --data_root for the data/output paths;
  * fixed seeds;
  * train and validation losses are appended to <log_path>/losses.csv (for the loss plots);
    TensorBoard logging is kept when tensorboard is installed;
  * --max_minutes: stop cleanly, saving a checkpoint, before a Kaggle session limit;
    --restore_step latest resumes from the newest checkpoint in ckpt_path;
  * checkpoints are written atomically (tmp file + rename).

Usage:
  python train.py --config ../English/config.yaml --data_root /kaggle/temp/en_data --max_minutes 690
"""
import argparse
import csv
import glob
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Dataset
from evaluate import evaluate
from model import FastSpeech2Loss
from utils.config import load_configs
from utils.model import get_model, get_param_num, get_vocoder
from utils.tools import to_device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSS_NAMES = ["total", "mel", "mel_postnet", "pitch", "energy", "duration"]


def latest_step(ckpt_path):
    steps = [int(os.path.basename(p).split(".")[0]) for p in glob.glob(os.path.join(ckpt_path, "*.pth.tar"))
             if os.path.basename(p).split(".")[0].isdigit()]
    return max(steps, default=0)


def save_checkpoint(model, optimizer, step, ckpt_path):
    path = os.path.join(ckpt_path, "{}.pth.tar".format(step))
    tmp = path + ".tmp"
    torch.save({"model": model.module.state_dict(), "optimizer": optimizer._optimizer.state_dict(), "step": step}, tmp)
    os.replace(tmp, path)
    return path


class LossLog:
    """Append-only CSV: split, step, lr, the six losses, wall-clock seconds."""

    def __init__(self, path):
        self.path = path
        new = not os.path.exists(path)
        self.f = open(path, "a", encoding="utf-8", newline="")
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(["split", "step", "lr"] + LOSS_NAMES + ["time_s"])

    def write(self, split, step, lr, losses, t):
        self.w.writerow([split, step, "{:.6g}".format(lr)] + ["{:.6f}".format(x) for x in losses] + ["{:.1f}".format(t)])
        self.f.flush()


def write_diagnostics(path, step, diag):
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["step"] + sorted(diag))
        w.writerow([step] + ["{:.6f}".format(diag[k]) for k in sorted(diag)])


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    configs = load_configs(args.config, args.data_root)
    preprocess_config, model_config, train_config = configs
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    ckpt_path, log_path = train_config["path"]["ckpt_path"], train_config["path"]["log_path"]
    if args.restore_step == "latest":
        args.restore_step = latest_step(ckpt_path)
    args.restore_step = int(args.restore_step)
    print("Prepare training ... (restore_step={}, device={})".format(args.restore_step, device), flush=True)

    # Get dataset (reference: sorted groups of batch_size * group_size utterances)
    dataset = Dataset("train.txt", preprocess_config, train_config, sort=True, drop_last=True)
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(dataset, batch_size=batch_size * group_size, shuffle=True,
                        collate_fn=dataset.collate_fn, num_workers=args.num_workers)

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param, flush=True)

    vocoder = get_vocoder(model_config, device) if args.tensorboard else None
    train_logger = val_logger = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        train_logger = SummaryWriter(os.path.join(log_path, "train"))
        val_logger = SummaryWriter(os.path.join(log_path, "val"))
    loss_log = LossLog(os.path.join(log_path, "losses.csv"))

    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    val_step = train_config["step"]["val_step"]

    t0 = time.time()
    deadline = t0 + args.max_minutes * 60 if args.max_minutes else None
    run_info = {"start_step": args.restore_step, "num_params": num_param, "device": str(device),
                "gpus": torch.cuda.device_count(), "dataset_size": len(dataset), "seed": args.seed}
    running = np.zeros(len(LOSS_NAMES))
    n_running = 0
    stop_reason = None

    outer_bar = tqdm(total=total_step, desc="Training", position=0, mininterval=60)
    outer_bar.n = args.restore_step
    outer_bar.update()
    while stop_reason is None:
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                # Forward
                output = model(*(batch[2:]))

                # Cal Loss
                losses = Loss(batch, output)
                total_loss = losses[0]

                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                running += np.array([l.item() for l in losses])
                n_running += 1
                lr = optimizer._optimizer.param_groups[0]["lr"]
                if not np.isfinite(running).all():
                    stop_reason = "non-finite loss at step {}".format(step)
                    break

                if step % log_step == 0:
                    means = running / n_running
                    loss_log.write("train", step, lr, means, time.time() - t0)
                    if train_logger is not None:
                        for name, v in zip(LOSS_NAMES, means):
                            train_logger.add_scalar("Loss/{}_loss".format(name), v, step)
                    outer_bar.write("Step {}, lr {:.3g}, ".format(step, lr) +
                                    ", ".join("{} {:.4f}".format(n, v) for n, v in zip(LOSS_NAMES, means)))
                    running[:] = 0
                    n_running = 0

                if step % val_step == 0:
                    model.eval()
                    diag = {}
                    message, val_means = evaluate(model, step, configs, val_logger, vocoder, return_losses=True,
                                                  diagnostics=diag)
                    loss_log.write("val", step, lr, val_means, time.time() - t0)
                    write_diagnostics(os.path.join(log_path, "val_diagnostics.csv"), step, diag)
                    message += ", Duration MAE {:.3f} frames, total-length rel. error {:.3f}".format(
                        diag["duration_mae_frames"], diag["total_length_rel_error"])
                    outer_bar.write(message)
                    model.train()

                out_of_time = deadline is not None and time.time() > deadline
                if step % save_step == 0 or step == total_step or out_of_time:
                    save_checkpoint(model, optimizer, step, ckpt_path)

                if step == total_step:
                    stop_reason = "total_step reached"
                    break
                if out_of_time:
                    stop_reason = "time budget ({} min) reached at step {}".format(args.max_minutes, step)
                    break
                step += 1
                outer_bar.update(1)
            if stop_reason is not None:
                break
            epoch += 1

    run_info.update({"end_step": step, "epochs_started": epoch, "stop_reason": stop_reason,
                     "minutes": round((time.time() - t0) / 60, 1)})
    with open(os.path.join(log_path, "run_{}_{}.json".format(args.restore_step, step)), "w") as f:
        json.dump(run_info, f, indent=1)
    print(json.dumps(run_info), flush=True)
    if stop_reason.startswith("non-finite"):
        raise SystemExit(stop_reason)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="English/config.yaml or Telugu/config.yaml")
    parser.add_argument("--data_root", type=str, default=None, help="root for the relative paths in config.yaml")
    parser.add_argument("--restore_step", default="0", help="step number, or 'latest'")
    parser.add_argument("--max_minutes", type=float, default=0, help="stop and checkpoint after this many minutes")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tensorboard", action="store_true", help="also log to TensorBoard (+ vocoded val samples)")
    main(parser.parse_args())
