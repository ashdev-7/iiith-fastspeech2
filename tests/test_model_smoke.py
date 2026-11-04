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
    # With a font name: exercises the shaped-image path (Pillow raqm) or the fallback path.
    from plotting import TITLE_RENDERING, shaped_text_image
    img = shaped_text_image("పులి అడ  విలో", "DejaVu Sans")
    assert img is None or (img.ndim == 3 and img.shape[2] == 4)
    paths = plot_variances(rec, str(tmp_path / "named"), telugu_font="DejaVu Sans")
    assert all(os.path.getsize(p) > 1000 for p in paths.values())
    assert TITLE_RENDERING["mode"].startswith(("Pillow raqm", "matplotlib text"))
    print("title rendering:", TITLE_RENDERING["mode"])
    csv_path = tmp_path / "losses.csv"
    csv_path.write_text("split,step,lr,total,mel,mel_postnet,pitch,energy,duration,time_s\n"
                        "train,100,1e-4,3,1,1,0.5,0.3,0.2,10\nval,100,1e-4,3.2,1,1,0.6,0.4,0.2,12\n")
    assert os.path.getsize(plot_losses([str(csv_path)], str(tmp_path / "loss.png"), "t")) > 1000
    # The assignment's file names: training_loss.png and validation_loss.png (+ overlay).
    import sys
    sys.path.insert(0, os.path.join(ROOT, "kaggle"))
    import pipeline
    out = pipeline.loss_plot(ROOT, [str(tmp_path)], str(tmp_path / "Results"), "t")
    assert [os.path.basename(p) for p in out] == ["training_loss.png", "validation_loss.png", "loss_curves.png"]
    assert all(os.path.getsize(p) > 1000 for p in out)


def test_pitch_energy_controls_scale_physical_units(tmp_path):
    """M12: p_control / e_control scale Hz (energy units), not the z-score. A voice below the corpus
    mean got LOWER with the reference's p * 1.2 (measured on the English test voice: 126 -> 113 Hz)."""
    from model.modules import VarianceAdaptor
    p = torch.tensor([-1.0, 0.5])  # stats in _configs: mean 180 Hz, std 40 -> 140 Hz, 200 Hz
    hz = lambda z: z * 40.0 + 180.0  # noqa: E731
    assert torch.allclose(hz(VarianceAdaptor.scale_physical(p, 1.2, 180.0, 40.0)), hz(p) * 1.2)
    assert VarianceAdaptor.scale_physical(p, 1.0, 180.0, 40.0) is p  # training / default: untouched
    torch.manual_seed(0)
    pre, mc, _ = _configs("en", tmp_path)
    model = FastSpeech2(pre, mc).to(device).eval()
    seq = torch.tensor([[5, 6, 7, 8, 9]], device=device)
    args = (torch.tensor([0], device=device), seq, torch.tensor([5], device=device), 5)
    with torch.no_grad():
        base, up = model(*args)[2], model(*args, p_control=1.2)[2]
        e_base, e_up = model(*args)[3], model(*args, e_control=0.8)[3]
    assert torch.allclose(hz(up), hz(base) * 1.2, atol=1e-3)
    assert torch.allclose(e_up * 20.0 + 30.0, (e_base * 20.0 + 30.0) * 0.8, atol=1e-3)


def test_per_phone_duration_control(tmp_path):
    """A (1, L) d_control tensor scales only the chosen phones (letter names in the English frontend)."""
    torch.manual_seed(0)
    pre, mc, _ = _configs("en", tmp_path)
    model = FastSpeech2(pre, mc).to(device).eval()
    seq = torch.tensor([[5, 6, 7, 8, 9]], device=device)
    args = (torch.tensor([0], device=device), seq, torch.tensor([5], device=device), 5)
    scale = torch.tensor([[1.0, 1.0, 1.6, 1.6, 1.0]], device=device)
    with torch.no_grad():
        base, scaled = model(*args), model(*args, d_control=scale)
    d0, d1 = base[5][0], scaled[5][0]
    assert torch.allclose(d1[[0, 1, 4]], d0[[0, 1, 4]])
    assert torch.allclose(d1[2:4], torch.round(d0[2:4] * 1.6))  # per-phone factors give whole frames
    assert int(scaled[9][0]) == int(sum(int(x) for x in d1))  # mel length = sum of the expanded durations


def test_per_phone_pitch_energy_controls_and_prosody_second_pass(tmp_path):
    """(1, L) p_control / e_control tensors scale only the chosen phones, in physical units; the prosody
    rules' second pass returns per-phone tensors that the model accepts."""
    import prosody_rules
    torch.manual_seed(0)
    pre, mc, _ = _configs("en", tmp_path)
    model = FastSpeech2(pre, mc).to(device).eval()
    with torch.no_grad():
        model.variance_adaptor.duration_predictor.linear_layer.bias.fill_(1.6)
    hz = lambda z: z * 40.0 + 180.0  # noqa: E731
    seq = torch.tensor([[5, 6, 7, 8, 9]], device=device)
    args = (torch.tensor([0], device=device), seq, torch.tensor([5], device=device), 5)
    pc = torch.tensor([[1.0, 1.1, 1.0, 0.9, 1.0]], device=device)
    with torch.no_grad():
        base, up = model(*args), model(*args, p_control=pc, e_control=pc)
    assert torch.allclose(hz(up[2]), hz(base[2]) * pc, atol=1e-3)
    assert torch.allclose(up[3] * 20.0 + 30.0, (base[3] * 20.0 + 30.0) * pc, atol=1e-3)
    phones = ["K", "AA1", "N", "D", "AH0"]
    words = [{"token": "w", "phones": phones}]
    p_c, e_c, d_c, log = prosody_rules.second_pass_controls(model, base, phones, words, "w.", "en", True, 1.0,
                                                            (1.0, 1.0, 1.0))
    assert p_c.shape == e_c.shape == d_c.shape == (1, 5)
    with torch.no_grad():
        out = model(*args, p_control=p_c, e_control=e_c, d_control=d_c)
    assert int(out[9][0]) == int(out[5][0].sum())


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


def test_train_checkpoint_loads_with_weights_only(tmp_path):
    """Checkpoint written by train.save_checkpoint after a real optimizer step must load with
    weights_only=True (the reference stored a numpy-scalar learning rate, which torch rejects)."""
    import torch.nn as nn
    from model import ScheduledOptim
    from train import save_checkpoint
    preprocess_config, model_config, train_config = _configs("te", tmp_path)
    model = nn.DataParallel(FastSpeech2(preprocess_config, model_config).to(device))
    opt = ScheduledOptim(model, train_config, model_config, 0)
    sum(p.sum() for p in model.parameters() if p.requires_grad).backward()
    opt.step_and_update_lr()
    path = save_checkpoint(model, opt, 7, str(tmp_path))
    ckpt = torch.load(path, map_location=device, weights_only=True)
    assert ckpt["step"] == 7 and isinstance(ckpt["optimizer"]["param_groups"][0]["lr"], float)
    resumed = ScheduledOptim(model, train_config, model_config, 7)
    resumed.load_state_dict(ckpt["optimizer"])


def test_vocoder_finetune_parts(tmp_path):
    """DiffMel reproduces the reference (non-differentiable) mel; discriminators run on a training segment;
    a fine-tuned generator checkpoint loads through get_vocoder."""
    import hifigan
    from audio.stft import TacotronSTFT
    from audio.tools import get_mel_from_wav
    from hifigan.discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator
    from utils.model import get_vocoder
    from vocoder_finetune import DiffMel
    stft = TacotronSTFT(1024, 256, 1024, 80, 22050, 0, 8000)
    rng = np.random.default_rng(0)
    y = (0.3 * np.sin(np.arange(8192) * 0.05) + 0.05 * rng.standard_normal(8192)).astype(np.float32)
    ref, _ = get_mel_from_wav(y, stft)
    ours = DiffMel(stft)(torch.from_numpy(y)[None])[0].detach().numpy()
    assert ref.shape == ours.shape
    assert np.abs(ref - ours).max() < 1e-3
    x = torch.randn(2, 1, 8192)
    for D in (MultiPeriodDiscriminator(), MultiScaleDiscriminator()):
        r, g, fr, fg = D(x, x * 0.5)
        assert len(r) == len(g) and all(torch.isfinite(t).all() for t in r)
    cfg = {"vocoder": {"model": "HiFi-GAN", "speaker": "universal"}}
    universal = get_vocoder(cfg, torch.device("cpu"))
    import json
    from utils.model import HIFIGAN_DIR
    G = hifigan.Generator(hifigan.AttrDict(json.load(open(os.path.join(HIFIGAN_DIR, "config.json")))))
    path = os.path.join(str(tmp_path), "g.pth.tar")
    torch.save({"generator": G.state_dict(), "step": 1}, path)
    ft = get_vocoder(cfg, torch.device("cpu"), path)
    assert type(ft) is type(universal)



def test_adapt_speaker_end_to_end(tmp_path):
    """Speaker adaptation on a tiny synthetic corpus: only the voice's train clips are used, the encoder
    stays frozen, validation metrics are finite and a loadable checkpoint is written."""
    import argparse
    import adapt_speaker
    from model import FastSpeech2 as FS2
    pre_cfg, mc, _ = _configs("te", tmp_path)
    pre = pre_cfg["path"]["preprocessed_path"]
    phones, _ = frontend("నమస్కారం అందరికీ", "te")
    text = "{" + " ".join(phones) + "}"
    rng = np.random.default_rng(0)
    for sub in ("mel", "pitch", "energy", "duration"):
        os.makedirs(os.path.join(pre, sub), exist_ok=True)
    lines = {"train.txt": [], "val.txt": []}
    for split, spk, name in [("train.txt", "spk_a", "a1"), ("train.txt", "spk_a", "a2"), ("train.txt", "spk_b", "b1"),
                             ("val.txt", "spk_a", "a3")]:
        d = np.full(len(phones), 4, dtype=np.int64)
        np.save(os.path.join(pre, "duration", "{}-duration-{}.npy".format(spk, name)), d)
        np.save(os.path.join(pre, "mel", "{}-mel-{}.npy".format(spk, name)), rng.normal(-5, 1, (int(d.sum()), 80)).astype(np.float32))
        np.save(os.path.join(pre, "pitch", "{}-pitch-{}.npy".format(spk, name)), rng.normal(0, 1, len(phones)))
        np.save(os.path.join(pre, "energy", "{}-energy-{}.npy".format(spk, name)), rng.normal(0, 1, len(phones)))
        lines[split].append("|".join([name, spk, text, "x"]))
    for f, ls in lines.items():
        open(os.path.join(pre, f), "w", encoding="utf-8").write("\n".join(ls) + "\n")
    torch.manual_seed(0)
    model = FS2(pre_cfg, mc)
    with torch.no_grad():
        model.variance_adaptor.duration_predictor.linear_layer.bias.fill_(1.6)
    ck = os.path.join(str(tmp_path), "base.pth.tar")
    torch.save({"model": model.state_dict(), "step": 5}, ck)
    enc0 = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    out = os.path.join(str(tmp_path), "adapt")
    adapt_speaker.main(argparse.Namespace(
        config=os.path.join(ROOT, CONFIG["te"]), data_root=str(tmp_path), checkpoint=ck, speaker="spk_a",
        out_dir=out, parts="decoder", steps=4, lr=1e-3, batch_size=2, eval_every=2, patience=5, seed=0))
    s = json.load(open(os.path.join(out, "adapt_summary.json")))
    assert s["train_clips"] == 2 and s["val_clips"] == 1
    assert all(np.isfinite(h["mcd_free"]) for h in s["history"])
    sd = torch.load(os.path.join(out, "adapted.pth.tar"), map_location="cpu", weights_only=True)["model"]
    for k, v in enc0.items():
        assert torch.equal(sd["encoder." + k].cpu(), v.cpu()), k
