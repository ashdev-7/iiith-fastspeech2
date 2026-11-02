# English checkpoint

The file is larger than GitHub's 100 MB per-file limit, so it is attached to a GitHub Release
(as the assignment allows: "an accessible external download link"):

| File | Size | SHA-256 |
|---|---|---|
| [`en_model.pth.tar`](https://github.com/ashdev-7/iiith-fastspeech2/releases/download/v1.0/en_model.pth.tar) | 135.3 MB | `b926c2d8b16bc855ee29d2eb5c0d3c7bdd10201d816bc67d2a9115da13bee3af` |

```bash
curl -L -o English/checkpoints/en_model.pth.tar \
  https://github.com/ashdev-7/iiith-fastspeech2/releases/download/v1.0/en_model.pth.tar
sha256sum English/checkpoints/en_model.pth.tar
```

Contents: `{"model": state_dict, "step": 17949}` (no optimizer state). Trained on Svarah (4.87 h) by
Kaggle job `fs2-03-train-en` (version 2), configuration `English/config.yaml`
(as run: `Results/English/train_config_used.yaml`). Load with
`torch.load(path, map_location=device, weights_only=True)["model"]`.
