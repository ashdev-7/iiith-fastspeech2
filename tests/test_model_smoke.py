"""Model smoke test for both languages (randomly initialised, CPU or GPU): the per-language vocabulary
sizes the embedding, every test sentence goes through frontend -> ids -> forward, one training step
gives finite losses, a checkpoint round-trip reproduces the outputs, and the plots render.
Needs torch, matplotlib and g2p_en; skipped where they are missing (runs on Kaggle)."""
import json
import os

import pytest

for _mod in ("numpy", "torch", "matplotlib", "g2p_en", "scipy"):
    pytest.importorskip(_mod)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from inference import frontend  # noqa: E402
from model import FastSpeech2, FastSpeech2Loss  # noqa: E402
from plotting import plot_losses, plot_variances  # noqa: E402
from text import text_to_sequence  # noqa: E402
from text.symbols import get_symbols  # noqa: E402
from utils.config import load_configs  # noqa: E402
from utils.tools import device  # noqa: E402  (masks are built on this device)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = {"en": "English/config.yaml", "te": "Telugu/config.yaml"}


def _configs(lang, tmp_path):
    configs = load_configs(os.path.join(ROOT, CONFIG[lang]), data_root=str(tmp_path))
    pre = configs[0]["path"]["preprocessed_path"]
    os.makedirs(pre, exist_ok=True)
    json.dump({"spk_a": 0, "spk_b": 1}, open(os.path.join(pre, "speakers.json"), "w"))
    json.dump({"pitch": [-2.0, 6.0, 180.0, 40.0], "energy": [-1.5, 5.0, 30.0, 20.0]},
              open(os.path.join(pre, "stats.json"), "w"))
    return configs


@pytest.mark.parametrize("lang", ["en", "te"])
def test_forward_train_step_and_checkpoint(lang, tmp_path, test_items):
    torch.manual_seed(0)
    preprocess_config, model_config, train_config = _configs(lang, tmp_path)
    model = FastSpeech2(preprocess_config, model_config).to(device)
    assert model.encoder.src_word_emb.num_embeddings == len(get_symbols(lang)) + 1

    # An untrained duration predictor outputs log_d ~ 0, i.e. 0 frames per phone; bias it to ~4 frames
    # so the inference path (length regulator, decoder) is exercised as after training.
    with torch.no_grad():
        model.variance_adaptor.duration_predictor.linear_layer.bias.fill_(1.6)
    model.eval()
    for item_id, item in test_items.items():
        if item["lang"] != lang:
            continue
        phones, _ = frontend(item["canonical"], lang)
        seq = text_to_sequence("{" + " ".join(phones) + "}", [], lang)
        assert len(seq) == len(phones) > 5, item_id
        with torch.no_grad():
            out = model(torch.tensor([1], device=device), torch.tensor([seq], device=device),
                            torch.tensor([len(seq)], device=device), len(seq))
        assert out[2].shape == out[3].shape == out[4].shape == (1, len(seq)), item_id
        assert int(out[5][0].sum()) == int(out[9][0]), item_id  # sum(d) == mel length

    # One training step on a fake batch with targets.
    model.train()
    T, L, B = 12, 40, 2
    texts = torch.randint(1, len(get_symbols(lang)), (B, T))
    src_lens = torch.tensor([T, T - 3])
    dur = torch.full((B, T), L // T)
    dur[1, T - 3:] = 0
    mel_lens = dur.sum(1)
    mels = torch.randn(B, int(mel_lens.max()), 80)
    pitch, energy = torch.randn(B, T), torch.randn(B, T)
    texts, src_lens, dur, mel_lens, mels, pitch, energy = (
        x.to(device) for x in (texts, src_lens, dur, mel_lens, mels, pitch, energy))
    batch = (["a", "b"], ["", ""], torch.tensor([0, 1], device=device), texts, src_lens, T, mels, mel_lens,
             int(mel_lens.max()), pitch, energy, dur)
    output = model(*batch[2:])
    losses = FastSpeech2Loss(preprocess_config, model_config)(batch, output)
    assert all(torch.isfinite(l) for l in losses)
    losses[0].backward()

    # Checkpoint round trip (same format as train.py) reproduces inference outputs.
    model.eval()
    path = tmp_path / "1.pth.tar"
    torch.save({"model": model.state_dict(), "optimizer": {}, "step": 1}, path)
    other = FastSpeech2(preprocess_config, model_config).to(device)
    other.load_state_dict(torch.load(path, map_location=device, weights_only=True)["model"])
    other.eval()
    with torch.no_grad():
        a = model(batch[2], texts, src_lens, T)[1]
        b = other(batch[2], texts, src_lens, T)[1]
    assert torch.equal(a, b)


def test_plots_render(tmp_path):
    rec = {"id": "T3", "original_text": "పులి అడ  విలో", "phones": ["p", "u", "l", "i", "a"],
           "duration_frames": [3, 5, 0, 4, 6], "frame_ms": 256 / 22.05, "pitch_hz": [180, 190, 0, 170, 160],
           "energy": [20, 30, 0, 25, 10], "generated_f0_hz": list(np.r_[np.zeros(3), np.full(15, 175.0)]),
           "generated_f0_period_ms": 256 / 22.05}
    paths = plot_variances(rec, str(tmp_path), telugu_font=None)
    assert all(os.path.getsize(p) > 1000 for p in paths.values())
    csv_path = tmp_path / "losses.csv"
    csv_path.write_text("split,step,lr,total,mel,mel_postnet,pitch,energy,duration,time_s\n"
                        "train,100,1e-4,3,1,1,0.5,0.3,0.2,10\nval,100,1e-4,3.2,1,1,0.6,0.4,0.2,12\n")
    assert os.path.getsize(plot_losses([str(csv_path)], str(tmp_path / "loss.png"), "t")) > 1000


def test_to_device_casts_float64_features():
    """Normalised pitch/energy .npy files are float64 (divided by float64 stats); the reference
    to_device left energy as double, which made backward() fail ("Found dtype Double")."""
    from utils.tools import to_device
    B, T, L = 2, 5, 9
    data = (["a", "b"], ["", ""], np.array([0, 1]), np.ones((B, T), dtype=np.int64), np.array([T, T]), T,
            np.zeros((B, L, 80), dtype=np.float32), np.array([L, L]), L,
            np.zeros((B, T), dtype=np.float64), np.zeros((B, T), dtype=np.float64), np.ones((B, T), dtype=np.int64))
    out = to_device(data, torch.device("cpu"))
    assert out[9].dtype == out[10].dtype == torch.float32
    assert out[11].dtype == torch.int64
