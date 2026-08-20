import os
import sys
import hashlib
import torch
import yaml

# Repo root (parent of scripts/) holds models.py and config.yaml.
# Run this script from the repo root: python scripts/generate_dummy_models.py
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from models import ConvAutoencoder, FeatureCNN, TemporalBiLSTM

def compute_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    models_dir = os.path.join(_REPO_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Instantiate models
    autoencoder = ConvAutoencoder()
    cnn = FeatureCNN()
    bilstm = TemporalBiLSTM()
    
    # Save dummy models
    ae_path = os.path.join(models_dir, "autoencoder.pt")
    cnn_path = os.path.join(models_dir, "cnn.pt")
    bilstm_path = os.path.join(models_dir, "bilstm.pt")
    
    torch.save(autoencoder.state_dict(), ae_path)
    torch.save(cnn.state_dict(), cnn_path)
    torch.save(bilstm.state_dict(), bilstm_path)
    
    # Compute hashes
    ae_hash = compute_sha256(ae_path)
    cnn_hash = compute_sha256(cnn_path)
    bilstm_hash = compute_sha256(bilstm_path)
    
    print(f"Generated models and calculated hashes:")
    print(f"Autoencoder: {ae_path} -> {ae_hash}")
    print(f"CNN: {cnn_path} -> {cnn_hash}")
    print(f"BiLSTM: {bilstm_path} -> {bilstm_hash}")
    
    # Update config.yaml
    config_path = os.path.join(_REPO_ROOT, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Store paths relative to the repo root using forward slashes so the config
    # is portable across Windows/Linux/Mac (scripts open these paths with the
    # cwd set to the repo root).
    config["models"] = {
        "autoencoder": {
            "path": "models/autoencoder.pt",
            "hash": ae_hash
        },
        "cnn": {
            "path": "models/cnn.pt",
            "hash": cnn_hash
        },
        "bilstm": {
            "path": "models/bilstm.pt",
            "hash": bilstm_hash
        }
    }
    
    # Add AI thresholds and parameters if not already present
    config["ai_sentry"] = {
        "anomaly_threshold": 0.5,
        "bilstm_confidence_threshold": 0.7,
        "hysteresis_window_size": 5,
        "hysteresis_required_confirmations": 3,
        "backpressure_max_queue_size": 50,
        "trigger_pub_port": 5557  # Port to publish confirmed target events
    }
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
        
    print("Updated config.yaml with model paths, hashes, and thresholds.")

if __name__ == "__main__":
    main()
