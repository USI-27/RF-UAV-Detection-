# Results

- `sample_audit_logs/` — example JSONL audit trails produced by
  `geometry.py`, `ai_sentry.py`, and `tracker.py` while running, kept here as
  a reference for the log schema each service writes. Fresh logs generated
  by a real run land at the repo root (`*_audit.jsonl`) and are git-ignored
  so working copies don't accumulate run history.
- Offline classifier metrics (confusion matrices, per-class F1, etc. for the
  BiLSTM+CNN+Autoencoder trained on CARDRF) belong here too once training is
  run — see the top-level README's [Results](../README.md#results) section
  for the target format.
