# Environment Setup

## Python environment (required for everything in this repo)

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Run all commands below from the repo root — every script resolves paths
(`config.yaml`, `models/`) relative to the current working directory.

## SDR / RF hardware (only needed for live captures, not for the simulated pipeline)

The real-time pipeline in this repo runs end-to-end using `ingestion.py`'s
built-in `SimulatedSDR`, so hardware is **not** required to develop against
or test the pipeline. It's needed if you want to feed it live RF instead of
simulated frames.

- **SDR++** — lightweight front-end for live spectrum visualization while
  you sanity-check a capture. https://www.sdrpp.org/
- **GNU Radio Companion** — block-based flow-graph tool, useful if you need
  to build a custom capture chain outside this repo's ingestion process.
  https://www.gnuradio.org/
- **Radioconda** — Conda-based distribution that bundles GNU Radio + drivers,
  easiest way to get a working GNU Radio install.
  https://github.com/radioconda/radioconda-installer
- **Zadig** — USB driver installer, needed on Windows for the SDR to enumerate
  correctly. https://zadig.akeo.ie/
- **Hardware**: HackRF One. Note the stock omnidirectional antenna cannot
  receive the standard FM broadcast band — that needs a dedicated antenna.

To wire a real device into `ingestion.py`, replace `SimulatedSDR` with a
SoapySDR- or HackRF-backed capture class that produces the same
`(num_antennas, frame_size_samples)` complex64 array shape the rest of the
pipeline expects (see `config.yaml`'s `sdr:` section for the exact
dimensions), and keep the same ZMQ publish format.

## Ports used by the pipeline

All bound to `127.0.0.1` by default (see `config.yaml` → `node.bind_address`).
Change this only if you understand the security implications — the ingestion
service's own test suite (`tests/test_ingestion.py`) explicitly checks that
these ports are *not* exposed on external interfaces.

| Port | Service | Purpose |
|---|---|---|
| 5555 | `ingestion.py` | Live I/Q frame publish (ZMQ PUB) |
| 5556 | `ingestion.py` | Historical buffer query (ZMQ REQ/REP) |
| 5558 | `geometry.py` | Azimuth/elevation estimate publish |
| 5557 | `ai_sentry.py` | Confirmed-target trigger publish |
| 5559 | `tracker.py` | 3D track publish |
| 8000 | `ingestion.py` | Prometheus-style `/metrics` HTTP endpoint |
