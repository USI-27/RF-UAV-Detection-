<div align="center">

# RF/SDR-Based UAV Detection & Localization

**Machine Learning–based passive detection and classification of unmanned aerial vehicles (UAVs) using RF/Software Defined Radio signals**

Summer Research Internship · Department of Electrical Engineering, IIT Ropar
Supervisor: Dr. Ashwani Sharma · Jun 2026 – Jul 2026

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Status](https://img.shields.io/badge/status-research--prototype-blue)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)

</div>

---

## Overview

Commercial drones are increasingly misused for unauthorized surveillance, smuggling, and border-security violations, particularly around sensitive installations. Conventional counter-UAV sensing — radar, acoustic, and optical/vision tracking — struggles against small, low radar-cross-section, low-altitude drones, and is expensive to deploy at scale.

This project builds a **passive RF-based detection pipeline**: nearly all commercial UAVs communicate with a ground controller over well-defined RF bands, so passively monitoring the spectrum with a Software Defined Radio (SDR) enables detection and classification **without the system ever transmitting a signal itself**.

| Detection Modality | Passive | Cost at Scale | Small/Low-RCS Drones | Range & All-Weather | Signal ID (Make/Model) |
|---|:---:|:---:|:---:|:---:|:---:|
| Radar | ✗ | ✗ | ✗ | ✓ | ✗ |
| Acoustic / Optical | △ | △ | △ | ✗ | ✗ |
| **RF / SDR (this project)** | ✓ | ✓ | ✓ | ✓ | ✓ |

This repository documents the RF signal-processing and ML pipeline built during the internship: dataset preprocessing and a hybrid deep-learning classifier for the RF/SDR detection modality above.

---

## Objectives

1. **SDR Proficiency** — hands-on fluency with SDR++, GNU Radio Companion, and the HackRF One SDR.
2. **Environment Setup** — install and configure the GNU Radio development stack; understand block-based RF signal-processing workflows.
3. **Dataset Preparation** — process the large-scale [CARDRF](#dataset) UAV RF dataset, converting raw `.mat` signal captures into waterfall spectrogram images.
4. **Model Design & Training** — design, implement, and train a hybrid deep-learning model (BiLSTM + CNN + Autoencoder) for RF-based UAV signal classification.
5. **Benchmarking** — benchmark the proposed architecture against baseline deep-learning models and analyze results.

---

## Results

| Metric | Value |
|---|---|
| Overall accuracy (17-class, held-out test set) | **85%** |
| Macro-avg F1 | 0.82 |
| Weighted-avg F1 | 0.84 |
| Test samples | 3,525 |
| Extended validation samples | 11,380 |
| Improvement over early 10-class baseline (47.1%) | **+38 points** |

**Strongest classes (F1 ≥ 0.97):** `MAVICPRO_HOVERING` (1.00), `IRIS` (1.00), `DJIM600_FLYING_NLOS` (0.99), `IPAD` (0.97), `BEEBEERUN` (0.97) — 13 of 17 classes reach F1 ≥ 0.82 overall.

**Known weaknesses:**
- `DJIM600` (idle state) — F1 = 0.00; all 64 idle-state samples misclassified (62/64 predicted as `INSPIRE`), suggesting near-identical idle/standby RF signatures between the two devices in the current feature representation.
- `PHANTOM_HOVERING` — Recall = 0.40; 72/131 samples confused with `INSPIRE_FLYING`, indicating operational-state (hovering vs. flying) is harder to separate than device identity in some cases.

Both findings were reproduced on an independent 11,380-sample extended validation batch (weighted P/R/F1: 0.86/0.81/0.79), confirming they are consistent, addressable model limitations rather than test artifacts. See [`results/`](results) for the full breakdown.

---

## Dataset

**[CARDRF](https://ieee-dataport.org/) (Cardinal RF)** — a large-scale RF dataset for UAV detection and classification research.

| | |
|---|---|
| Raw data size | 68 GB (`.mat` I/Q matrices) |
| Total RF classes | 17 |
| Background / everyday RF emitters | 5 — Fitbit, iPad, iPhone, Motorola, WiFi |
| UAV RF-signal classes | 12 — across idle, flying, hovering, and non-line-of-sight (NLOS) states: `BEEBEERUN`, `DJIM600`, `DJIM600_FLYING`, `DJIM600_FLYING_NLOS`, `INSPIRE`, `INSPIRE_FLYING`, `INSPIRE_FLYING_NLOS`, `MAVICPRO`, `MAVICPRO_HOVERING`, `PHANTOM_FLYING`, `PHANTOM_HOVERING`, `IRIS` |

> The dataset is **not included** in this repository due to its size (68 GB). See [`data/README.md`](data/README.md) for access and preprocessing instructions.

### Preprocessing pipeline

Raw I/Q captures are converted into 2D time-frequency waterfall images via Short-Time Fourier Transform (STFT), preserving burst timing and frequency-hopping structure as learnable spatial texture.

```
Raw .mat (I/Q matrices) → STFT → Normalization & windowing → 400×400 PNG waterfall chunks
```

The 68 GB dataset was split into 10 batches and distributed across 10 lab machines, each running an identical Python conversion script in parallel, with resulting waterfall images collected back onto the main workstation. This reduced processing time from an estimated **3+ days (single machine)** to **~1 day**. See [`src/preprocessing/`](src/preprocessing).

---

## Model Architecture

A hybrid model fusing spatial texture, temporal sequence, and denoised latent features:

- **CNN branch** — extracts local time-frequency texture from waterfall images: burst patterns, frequency-hopping structure, modulation-specific spectral shapes.
- **BiLSTM branch** — models sequential dependencies across the time axis in both directions; captures FHSS frequency-hopping and intermittent burst patterns of UAV control links.
- **Autoencoder** — unsupervised feature compression and denoising prior to classification; improves robustness to background RF noise and reduces dimensionality.

The fused feature representation feeds a fully-connected softmax classification head producing a probability distribution over all 17 target classes.

```
Incoming RF capture (HackRF One)
        │
        ▼
      STFT ──► Waterfall spectrogram
        │
   ┌────┴─────────────┬──────────────────┐
   ▼                   ▼                  ▼
CNN branch        BiLSTM branch      Autoencoder
(spatial texture) (temporal seq.)    (denoising)
   └────────┬──────────┴──────────────────┘
            ▼
   Fused feature representation
            ▼
  FC Softmax Classification Head → 17-class probability distribution
```

Trained continuously for ~5 days; trained weights are persisted for reuse without retraining. See [`src/models/`](src/models).

### Benchmark against baselines

| Approach | Scope | Accuracy | Key characteristic |
|---|---|---|---|
| Early baseline classifier | 10-class pilot subset | 47.1% | High cross-class confusion; no temporal modeling |
| Autoencoder + 2D CNN (open-set variant)* | 4 classes + novelty layer | 83%+ | Strong unknown-hardware / anomaly detection; limited class granularity |
| **BiLSTM + CNN + Autoencoder (this work)** | 17 classes (full CARDRF) | **85%** | Spatial + temporal fusion; best generalization across full taxonomy |

<sub>*From a related open-set exploration on a 4-class DJI/BeeBeeRun subset of 2.4 GHz ISM-band captures — a design comparison point, not a direct ablation of the proposed model.</sub>

---

## Robustness to Adversarial Evasion

| Evasion technique | Description | Countermeasure |
|---|---|---|
| Frequency-hopping evasion | Agile drones hop channels to break narrowband detectors | Wideband captures with extended STFT windows let the model learn the full hop-pattern sequence, not a single carrier |
| Protocol mimicry (Wi-Fi trap) | Drones disguise control signals to mimic Wi-Fi traffic | Open-set anomaly layer computes embedding distances; DBSCAN clustering isolates "suspicious / unknown threat" clusters |

---

## Tech Stack & Hardware

**Software:** [SDR++](https://www.sdrpp.org/) (lightweight SDR front-end for live spectrum visualization), [GNU Radio Companion](https://www.gnuradio.org/) (block-based RF signal-processing flow-graph tool), [Radioconda](https://github.com/radioconda/radioconda-installer) (Conda-based GNU Radio distribution), [Zadig](https://zadig.akeo.ie/) (USB driver installer for the SDR)

**Hardware:** [HackRF One](https://greatscottgadgets.com/hackrf/) SDR (stock omnidirectional antenna — note: cannot receive standard FM band without a dedicated antenna)

**Training workstation:** Intel i9 (14th gen), 96 GB RAM, 32 GB GPU VRAM, 4 TB storage

**ML stack:** Python, PyTorch/TensorFlow (see [`requirements.txt`](requirements.txt))

> A shared institutional GPU server was evaluated for accelerated training but could not be used successfully within the internship timeframe due to persistent CUDA library configuration issues. Final training was carried out entirely on the dedicated lab workstation.

---

## Repository Structure

```
RF-UAV-Detection/
├── README.md                    # This file
├── LICENSE
├── requirements.txt
├── config.yaml                  # Ports, node positions, thresholds, model paths — shared by every service below
├── ingestion.py                 # Node-local SDR capture (simulated) + rolling buffer + /metrics endpoint
├── buffer_client.py             # Client used by every downstream service to talk to ingestion.py
├── geometry.py                  # Per-node MUSIC angle-of-arrival engine
├── models.py                    # ConvAutoencoder + FeatureCNN + TemporalBiLSTM definitions
├── ai_sentry.py                 # Runs the trained models over live frames, publishes confirmed-target triggers
├── ekf.py                       # Extended Kalman Filter for single-track 3D state estimation
├── tracker.py                   # Multi-node triangulation + track association + EKF lifecycle
├── docs/
│   ├── environment_setup.md     # SDR++ / GNU Radio / HackRF One setup guide + service port table
│   ├── architecture.md          # Real-time pipeline data flow, file-by-file guide, known gaps
│   └── report.pdf               # Full internship report (optional)
├── data/
│   └── README.md                # CARDRF dataset access & structure notes
├── models/                      # Model weights (.pt) + SHA-256 hashes referenced by config.yaml
├── scripts/
│   ├── run_pipeline.py          # Launches ingestion/replay + geometry + ai_sentry + tracker together
│   ├── generate_dummy_models.py # Produces placeholder weights for exercising the pipeline pre-training
│   ├── play_dataset.py          # Replays pre-processed .npy captures on the ingestion port
│   └── play_mat_dataset.py      # Replays raw CARDRF .mat captures on the ingestion port
├── tests/                       # unittest suite: geometry, tracker+EKF, ai_sentry, ingestion (integration)
├── notebooks/                   # Exploratory analysis notebooks
└── results/
    └── sample_audit_logs/       # Example JSONL audit trails from geometry/ai_sentry/tracker
```

See [`docs/architecture.md`](docs/architecture.md) for how the real-time
detection & localization pipeline (the code at the repo root) fits together —
it's the deployment-time counterpart to the offline classifier described
above, running the same BiLSTM+CNN+Autoencoder model against a live stream
and fusing bearings from multiple nodes into 3D tracks.

---

## Getting Started

```bash
git clone https://github.com/USI-27/RF-UAV-Detection-.git
cd RF-UAV-Detection-
pip install -r requirements.txt
```

For SDR hardware and GNU Radio environment setup, see [`docs/environment_setup.md`](docs/environment_setup.md).
For dataset access and preprocessing, see [`data/README.md`](data/README.md).

**Real-time detection & localization pipeline** (implemented — see [`docs/architecture.md`](docs/architecture.md)):

```bash
# Run the whole pipeline (simulated SDR by default)
python scripts/run_pipeline.py config.yaml

# ...or run each service standalone
python ingestion.py config.yaml
python geometry.py config.yaml
python ai_sentry.py config.yaml
python tracker.py config.yaml
```

**Offline classifier training** (preprocessing/training/evaluation scripts not yet in this repo — the model
architecture in `models.py` matches the design below, `ai_sentry.py` is the inference-time consumer of the
trained weights):

```bash
# Convert raw .mat captures to waterfall spectrograms
python src/preprocessing/mat_to_waterfall.py --input <path_to_cardrf> --output data/processed/

# Train the hybrid model
python src/training/train.py --config src/training/config.yaml

# Evaluate on held-out test set
python src/evaluation/evaluate.py --weights results/model_weights.pt
```

---

## Future Work

- [ ] Correct the complete `DJIM600` misclassification via additional samples, class rebalancing, or feature engineering.
- [ ] Improve `PHANTOM_HOVERING` recall by capturing operational-state features beyond device identity.
- [ ] Resolve the CUDA / GPU-server configuration issue to enable faster iteration on larger models.
- [ ] Extend the pipeline toward multi-receiver / direction-finding for full localization.
- [ ] Evaluate open-set performance — flagging genuinely unknown UAV signal types beyond the 17 known classes.
- [ ] Field-test the trained model on live SDR captures using the HackRF One setup.

---

## Acknowledgements

This work was carried out during a Summer Research Internship in the Department of Electrical Engineering, IIT Ropar, under the supervision of **Dr. Ashwani Sharma**.

## Author

**Udbhav Singh**
B.Tech, Electronics and Communication Engineering, The LNM Institute of Information Technology, Jaipur
[LinkedIn](https://www.linkedin.com/in/udbhavsingh27/) · [GitHub](https://github.com/USI-27) · udbhavsingh27@gmail.com

## License

This project is licensed under the [MIT License](LICENSE).
