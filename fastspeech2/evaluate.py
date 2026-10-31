import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.config import load_configs
from utils.model import get_model, get_vocoder
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, step, configs, logger=None, vocoder=None, return_losses=False, diagnostics=None):
    """diagnostics: optional dict, filled with duration-predictor metrics on the validation set
    (rounded predicted vs. MFA durations, per valid phone; total-length error per utterance)."""
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)

    # Evaluation
    loss_sums = [0 for _ in range(6)]
    dur_abs_err, n_phones, rel_len_err, n_utts = 0.0, 0, 0.0, 0
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                # Forward
                output = model(*(batch[2:]))

                # Cal Loss
                losses = Loss(batch, output)

                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])

                # Duration diagnostics (reference conversion: clamp(round(exp(log_d) - 1), 0)).
                valid = ~output[6]
                pred_d = torch.clamp(torch.round(torch.exp(output[4]) - 1), min=0) * valid
                true_d = batch[11].float() * valid
                dur_abs_err += (pred_d - true_d).abs().sum().item()
                n_phones += int(valid.sum().item())
                tot_true = true_d.sum(1).clamp(min=1)
                rel_len_err += ((pred_d.sum(1) - true_d.sum(1)).abs() / tot_true).sum().item()
                n_utts += len(batch[0])

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]
    if diagnostics is not None:
        diagnostics["duration_mae_frames"] = dur_abs_err / max(1, n_phones)
        diagnostics["total_length_rel_error"] = rel_len_err / max(1, n_utts)

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
        *([step] + [l for l in loss_means])
    )

    if logger is not None:
        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            batch,
            output,
            vocoder,
            model_config,
            preprocess_config,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )

    if return_losses:
        return message, loss_means
    return message


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=30000)
    parser.add_argument("--config", type=str, required=True, help="English/config.yaml or Telugu/config.yaml")
    parser.add_argument("--data_root", type=str, default=None)
    args = parser.parse_args()

    # Read Config
    configs = load_configs(args.config, args.data_root)

    # Get model
    model = get_model(args, configs, device, train=False).to(device)

    message = evaluate(model, args.restore_step, configs)
    print(message)