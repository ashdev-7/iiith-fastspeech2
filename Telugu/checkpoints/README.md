# Telugu checkpoint

The file is larger than GitHub's 100 MB per-file limit, so it is attached to a GitHub Release
(as the assignment allows: "an accessible external download link"):

| File | Size | SHA-256 |
|---|---|---|
| [`te_model.pth.tar`](https://github.com/ashdev-7/iiith-fastspeech2/releases/download/v1.0/te_model.pth.tar) | 134.9 MB | `b72501d7a8c83ee79ebf2d72a6f33fb9fc648b304590c76861ff85cbe70ec546` |

```bash
curl -L -o Telugu/checkpoints/te_model.pth.tar \
  https://github.com/ashdev-7/iiith-fastspeech2/releases/download/v1.0/te_model.pth.tar
sha256sum Telugu/checkpoints/te_model.pth.tar
```

Contents: `{"model": state_dict, "step": 55524}` (no optimizer state). Trained on IndicTTS Telugu
(4.96 h selected) by Kaggle job `fs2-03-train-te`, configuration `Telugu/config.yaml`
(as run: `Results/Telugu/train_config_used.yaml`). Load with
`torch.load(path, map_location=device, weights_only=True)["model"]`.
