# Real-Time Detection & Localization Pipeline

This document describes the code in the repo root (`ingestion.py`, `geometry.py`,
`ai_sentry.py`, `tracker.py`, `ekf.py`, `models.py`, `buffer_client.py`). It is
the deployment-time counterpart to the offline BiLSTM+CNN+Autoencoder
classifier described in the main README: instead of training on a static
CARDRF batch, it runs the trained model as a live "sentry" watching a
continuous SDR stream, then fuses angle-of-arrival estimates from multiple
nodes into 3D tracks — the "localization" half of the project.

## Data flow

```
                 ┌────────────────────┐
                 │  ingestion.py       │  SimulatedSDR (stand-in for a HackRF/SoapySDR
                 │  (per node)         │  device) generates coherent 5-antenna I/Q frames.
                 │  pub_port: 5555     │  Buffer_Management_Process keeps a rolling history
                 │  rep_port: 5556     │  for random-access replay queries.
                 └─────────┬───────────┘
                           │ ZMQ PUB (live frames)      ▲ ZMQ REQ/REP (history queries)
                           ▼                            │
                 ┌────────────────────┐         buffer_client.py
                 │  geometry.py        │  MUSIC direction-finding over a 5-element
                 │  angles_pub_port:   │  uniform circular array: steering vectors,
                 │  5558                │  forward-backward spatial smoothing (FBSS),
                 └─────────┬───────────┘  covariance regularization.
                           │ ZMQ PUB (az/el estimates)
                           ├─────────────────────────────┐
                           ▼                              ▼
                 ┌────────────────────┐         ┌────────────────────┐
                 │  ai_sentry.py       │         │  tracker.py         │
                 │  trigger_pub_port:  │         │  tracks_pub_port:   │
                 │  5557                │         │  5559                │
                 └────────────────────┘         └────────────────────┘
                 CNN embeds each frame,          Combines az/el bearings from
                 BiLSTM checks the embedding     >=2 nodes via least-squares
                 sequence for FHSS hop patterns, triangulation, gates candidate
                 autoencoder reconstruction       measurements against existing
                 error flags background-RF        tracks (chi-squared + Hungarian
                 anomalies. Hysteresis requires    assignment), and runs a 3D
                 N consecutive confirmations       constant-velocity EKF per
                 before raising a trigger.         confirmed track (birth/death
                                                    counters prevent flicker).
```

Every service is an independent OS process that only talks to the others over
ZMQ (see `config.yaml` for the ports above) — there's no shared Python state.
That's why each file has a `if __name__ == "__main__":` block that takes a
`config_path` argument and runs forever until Ctrl+C. `scripts/run_pipeline.py`
starts all of them together for convenience.

## File-by-file

| File | Role |
|---|---|
| `ingestion.py` | Node-local SDR capture (simulated) + rolling buffer + Prometheus-style `/metrics` HTTP endpoint. Runs as two subprocesses (ingest + buffer manager). |
| `buffer_client.py` | Thin client used by every downstream service to subscribe to live frames or query buffered history from a node's ingestion process. |
| `geometry.py` | Per-node MUSIC angle-of-arrival engine over the 5-antenna UCA. Publishes azimuth/elevation estimates. |
| `models.py` | The three PyTorch model definitions: `ConvAutoencoder` (denoising/anomaly), `FeatureCNN` (spatial embedding), `TemporalBiLSTM` (temporal confidence over a sequence of embeddings). Same architecture as the offline classifier described in the top-level README. |
| `ai_sentry.py` | Loads the three models (with a SHA-256 integrity check against `config.yaml`), runs them over live frames, and publishes confirmed-target trigger events after hysteresis. |
| `ekf.py` | Extended Kalman Filter for a single 3D constant-velocity track (state: position + velocity). |
| `tracker.py` | Multi-node triangulation, track association (Hungarian algorithm + chi-squared gating), and per-track EKF lifecycle (birth/death counters). |
| `config.yaml` | Single source of truth for ports, node positions, thresholds, and model paths/hashes. Every service takes this file's path as its only CLI argument. |
| `scripts/generate_dummy_models.py` | Produces randomly-initialized weights for the three models and updates `config.yaml` with matching hashes — useful for exercising the pipeline before real trained weights exist. **The weights currently committed under `models/` were produced this way; they are not trained on CARDRF yet.** |
| `scripts/play_dataset.py` | Replays pre-processed `.npy` I/Q captures on the ingestion port, at real-time pacing, in place of the simulated SDR. |
| `scripts/play_mat_dataset.py` | Same idea, but reads raw CARDRF `.mat` captures directly (via `scipy.io.loadmat`) instead of pre-converted `.npy` files. |
| `scripts/run_pipeline.py` | New convenience launcher — starts ingestion/replay + geometry + ai_sentry + tracker together and shuts them all down on Ctrl+C. |
| `tests/` | `unittest`-based tests. `test_geometry.py`, `test_tracker.py`, `test_ai_sentry.py` exercise the algorithms directly; `test_ingestion.py` spawns `ingestion.py` as a subprocess and drives it over ZMQ (real end-to-end integration test, not a unit test — it needs the ports in `config.yaml` free). |
| `results/sample_audit_logs/` | Example JSONL audit trails (`geometry_audit.jsonl`, `ai_sentry_audit.jsonl`, `tracker_assignment_audit.jsonl`) each service appends to at runtime, kept here as illustrative samples. Fresh logs generated during a real run are git-ignored (see `.gitignore`). |

## Running it

From the repo root, with the dummy/current weights:

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py config.yaml
```

Each service also runs standalone, which is how the test suite drives them:

```bash
python ingestion.py config.yaml
python geometry.py config.yaml
python ai_sentry.py config.yaml
python tracker.py config.yaml
```

To replay a recorded dataset instead of the simulated SDR:

```bash
python scripts/run_pipeline.py config.yaml --replay-mat data/raw/some_session/
```

## Known gaps / next steps

- `config.yaml` currently only defines one node (`Node_A`) as the process
  that runs; `Node_B`'s position is defined under `tracking.node_positions`
  for triangulation math, but there's no second `ingestion.py`/`geometry.py`
  instance wired up yet. Running the pipeline with two physical/simulated
  nodes requires a second config (different ports + `node.id: Node_B`) and a
  second set of `ingestion.py`/`geometry.py` processes.
- The committed model weights are untrained placeholders from
  `scripts/generate_dummy_models.py`. Swap in real CARDRF-trained weights and
  regenerate the hashes in `config.yaml` before relying on `ai_sentry.py`'s
  detections.
- No training/evaluation script for the classifier is included in this drop —
  see the top-level README's dataset section for the offline training side of
  the project.
