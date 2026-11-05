"""clarity.py: micro-pause insertion, duration factors, edges and mel denoising (torch, no model)."""
import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
import clarity as C  # noqa: E402
import prosody_rules as R  # noqa: E402

PHONES = ["S", "EH1", "D", "DH", "AH0", "P", "AA2", "R", "L", "AH0", "M", "EH1", "N", "T", "ER0", "IY0", "sp", "N", "OW1"]
TRACE = {"words": [{"token": "said", "phones": ["S", "EH1", "D"]}, {"token": "the", "phones": ["DH", "AH0"]},
                   {"token": "parliamentary", "phones": PHONES[5:16]},
                   {"token": ",", "phones": ["sp"]}, {"token": "no", "phones": ["N", "OW1"]}],
         "duration_scale": [1.0] * 19}


def test_prepare_pauses_before_the_function_word_and_scales_long_words():
    phones, trace, f = C.prepare(PHONES, TRACE, "en", long_word_scale=1.25, heavy_pause=True)
    assert phones[3] == "sp" and phones[4:6] == ["DH", "AH0"] and len(phones) == len(PHONES) + 1
    assert [w["token"] for w in trace["words"]][:4] == ["said", "<pause>", "the", "parliamentary"]
    assert len(f) == len(phones) and np.all(f[6:17] == 1.25) and np.all(f[:6] == 1)
    assert R.align_words(phones, trace["words"])[2][1] == list(range(6, 17))   # still aligned
    start = {"words": TRACE["words"][1:], "duration_scale": [1.0] * 16}        # long word at sentence start:
    assert C.prepare(PHONES[3:], start, "en", heavy_pause=True)[0] == PHONES[3:]       # no pause
    same, _, f2 = C.prepare(PHONES, TRACE, "en")                                 # defaults: no change
    assert same == PHONES and np.all(f2 == 1)


def test_edges_fade_to_silence():
    mel = torch.full((20, 80), -5.0)
    out, lead = C.edges(mel)
    assert lead == 4 and out.shape[0] == 20 + 4 + 5 + 10
    assert torch.all(out[:4] == C.SIL) and torch.all(out[-10:] == C.SIL)
    fade = out[24:29, 0]
    assert torch.all(fade[1:] < fade[:-1]) and fade[0] < -5.0 and fade[-1] > C.SIL


def test_denoise_removes_the_floor_and_keeps_speech():
    prof = np.full(80, -6.5, dtype=np.float32)
    mel = torch.full((30, 80), -6.5)
    mel[10:20] = -1.0                                      # strong speech frames (SNR ~ 48 dB)
    out = C.denoise(mel, prof)
    assert torch.all(out[:5] <= -6.5 + np.log(0.1) + 1e-4)            # floor frames: -20 dB (gain floor)
    assert torch.allclose(out[13:17], mel[13:17], atol=1e-3)          # speech untouched
