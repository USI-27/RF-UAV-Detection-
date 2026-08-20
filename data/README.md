# Data

The CARDRF dataset (68 GB of raw `.mat` I/Q captures across 17 RF-emitter
classes) is not committed to this repo — see the top-level README's
[Dataset](../README.md#dataset) section for what it contains and where to
request access.

## Expected layout

If/when you download it, keep it out of git (already covered by
`.gitignore`) and organize it as:

```
data/
├── raw/            # original CARDRF .mat captures, one subfolder per session/class
└── processed/      # STFT waterfall images or .npy frames derived from raw/
```

## Feeding data into the pipeline

Two scripts in `scripts/` replay data on the same ZMQ port the simulated SDR
would use, so the rest of the pipeline (`geometry.py`, `ai_sentry.py`,
`tracker.py`) doesn't need to know the difference:

- `scripts/play_mat_dataset.py <config.yaml> <folder>` — streams raw
  `data/raw/...` `.mat` captures directly.
- `scripts/play_dataset.py <config.yaml> <folder>` — streams pre-processed
  `.npy` frames from `data/processed/...`.

Both pace playback to match the configured `sample_rate_hz` / `frame_size_samples`
so downstream timing-sensitive logic (e.g. the BiLSTM's hop-pattern window)
sees realistic frame spacing.
