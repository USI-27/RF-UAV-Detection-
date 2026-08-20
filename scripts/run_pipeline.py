"""
run_pipeline.py — convenience launcher for the full real-time pipeline.

Starts, in order: ingestion -> geometry -> ai_sentry -> tracker, each as its
own subprocess (matching how they're designed to run: independent services
talking over ZMQ, see config.yaml for ports). Ctrl+C stops all of them.

Usage (run from the repo root):
    python scripts/run_pipeline.py config.yaml

Optional: stream a recorded dataset instead of the simulated SDR by passing
--replay <dataset_folder> (uses scripts/play_dataset.py, expects .npy files)
or --replay-mat <dataset_folder> (uses scripts/play_mat_dataset.py, expects
raw CARDRF .mat captures). When a replay flag is given, ingestion.py is not
started — the replay script publishes frames on the same port instead.
"""

import argparse
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Launch the RF-UAV detection & localization pipeline.")
    parser.add_argument("config_path", nargs="?", default="config.yaml",
                         help="Path to config.yaml (default: config.yaml in repo root)")
    parser.add_argument("--replay", metavar="DATASET_FOLDER", default=None,
                         help="Stream .npy frames from this folder instead of the simulated SDR")
    parser.add_argument("--replay-mat", metavar="DATASET_FOLDER", default=None,
                         help="Stream raw CARDRF .mat frames from this folder instead of the simulated SDR")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config_path)
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    services = []

    def spawn(name, script_rel_path, extra_args=None):
        script_path = os.path.join(REPO_ROOT, script_rel_path)
        cmd = [sys.executable, script_path, config_path] + (extra_args or [])
        print(f"Starting {name}: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT)
        services.append((name, proc))
        return proc

    try:
        if args.replay_mat:
            spawn("dataset-replay (.mat)", "scripts/play_mat_dataset.py", [args.replay_mat])
        elif args.replay:
            spawn("dataset-replay (.npy)", "scripts/play_dataset.py", [args.replay])
        else:
            spawn("ingestion", "ingestion.py")
        time.sleep(1.5)  # let the SDR/replay publisher bind before downstream services connect

        spawn("geometry", "geometry.py")
        spawn("ai_sentry", "ai_sentry.py")
        spawn("tracker", "tracker.py")

        print("\nAll services started. Press Ctrl+C to stop the pipeline.\n")
        while True:
            time.sleep(1)
            for name, proc in services:
                ret = proc.poll()
                if ret is not None:
                    print(f"[{name}] exited early with code {ret} — stopping remaining services.")
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\nShutting down pipeline...")
    finally:
        for name, proc in reversed(services):
            if proc.poll() is None:
                proc.terminate()
        for name, proc in reversed(services):
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("All services stopped.")


if __name__ == "__main__":
    main()
