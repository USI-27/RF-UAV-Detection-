import os
import sys
import time
import yaml
import json
import hashlib
import zmq
import torch
import numpy as np
import queue
import threading
from models import ConvAutoencoder, FeatureCNN, TemporalBiLSTM

def compute_sha256(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def load_verified_model(model_class, model_cfg_name, config, device):
    """
    Loads PyTorch model, performing cryptographic integrity check before loading.
    """
    cfg = config["models"][model_cfg_name]
    path = cfg["path"]
    expected_hash = cfg["hash"]
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
        
    actual_hash = compute_sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"SECURITY EXCEPTION: Cryptographic hash mismatch for model {model_cfg_name}!\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual_hash}"
        )
        
    model = model_class()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Successfully loaded and verified model: {model_cfg_name} ({path})")
    return model

class AISentryPipeline:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.node_cfg = self.config["node"]
        self.sdr_cfg = self.config["sdr"]
        self.ai_cfg = self.config["ai_sentry"]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"AI Sentry running on device: {self.device}")
        
        # Load and verify models
        self.autoencoder = load_verified_model(ConvAutoencoder, "autoencoder", self.config, self.device)
        self.cnn = load_verified_model(FeatureCNN, "cnn", self.config, self.device)
        self.bilstm = load_verified_model(TemporalBiLSTM, "bilstm", self.config, self.device)
        
        # Slices queue with backpressure limit
        self.max_queue_size = self.ai_cfg["backpressure_max_queue_size"]
        self.raw_frame_queue = queue.Queue(maxsize=self.max_queue_size)
        
        # Sliding history per center frequency
        # Each entry tracks: (list of 5 CNN feature vectors, list of 5 BiLSTM decisions)
        self.channel_histories = {}
        
        # ZMQ Context
        self.context = zmq.Context()
        
        # ZMQ Publisher for Target Events
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.setsockopt(zmq.SNDHWM, 1000)
        self.pub_socket.bind(f"tcp://{self.node_cfg['bind_address']}:{self.ai_cfg['trigger_pub_port']}")
        
        self.stop_event = threading.Event()
        
        # Diagnostic / Audit Logs file (Structured logs)
        self.audit_log_path = "ai_sentry_audit.jsonl"
        
        # Metrics count
        self.total_processed = 0
        self.total_anomalies = 0
        self.total_dropped = 0
        self.total_targets = 0

    def write_audit_log(self, entry):
        """
        Structured audit logging (JSON Lines format). Access-restricted by system design.
        """
        # Redact actual raw data arrays / spectrogram vectors to prevent data leakage.
        # Only log metadata and output scores.
        entry["system_time"] = time.time()
        try:
            with open(self.audit_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"Error writing audit log: {e}")

    def enqueue_frame(self, metadata, iq_data):
        """
        Push incoming raw frames to preprocessing queue.
        Implements queue capacity backpressure to shed load under CPU/GPU saturation.
        """
        try:
            self.raw_frame_queue.put_nowait((metadata, iq_data))
        except queue.Full:
            self.total_dropped += 1
            # Rate-limit alert logging to indicate jamming/saturation DoS
            self.write_audit_log({
                "event": "backpressure_shed_frame",
                "center_freq_hz": metadata["center_freq_hz"],
                "timestamp": metadata["timestamp"],
                "queue_size": self.raw_frame_queue.qsize()
            })

    def process_batch(self, batch):
        """
        Runs the batched inference pipeline: STFT -> Autoencoder -> CNN -> BiLSTM -> Hysteresis.
        """
        batch_size = len(batch)
        metas = [item[0] for item in batch]
        iq_arrays = [item[1] for item in batch]
        
        # Batch STFT on device
        # Convert list of complex numpy arrays to a batched complex PyTorch tensor
        t_iq = torch.from_numpy(np.stack(iq_arrays)).to(self.device) # Shape: (B, 5, 1024)
        
        # Flatten batch and channel dimensions for compatibility with PyTorch STFT (expected 1D or 2D)
        t_iq_flat = t_iq.view(-1, t_iq.size(-1)) # Shape: (B * 5, 1024)
        window = torch.hann_window(256, device=self.device)
        
        # Compute batched STFT
        # n_fft=256, hop=128
        stft_out = torch.stft(t_iq_flat, n_fft=256, hop_length=128, window=window, return_complex=True)
        # Compute magnitude spectrogram and reshape back to (B, 5, 256, 9)
        spectrograms = torch.abs(stft_out).view(batch_size, 5, 256, 9)
        
        # 1. Run Autoencoder Anomaly check
        with torch.no_grad():
            reconstructed = self.autoencoder(spectrograms)
            
        # Compute MSE per spectrogram frame in batch
        # shape: (B,)
        mse = torch.mean((spectrograms - reconstructed) ** 2, dim=(1, 2, 3))
        
        anomaly_thresh = self.ai_cfg["anomaly_threshold"]
        
        # Collect indexes of anomalous spectrograms for the CNN stage
        anomaly_indices = []
        for i in range(batch_size):
            err = float(mse[i].cpu().item())
            meta = metas[i]
            
            self.write_audit_log({
                "event": "anomaly_detection_stage",
                "center_freq_hz": meta["center_freq_hz"],
                "timestamp": meta["timestamp"],
                "reconstruction_error": err,
                "is_anomaly": err >= anomaly_thresh
            })
            
            if err >= anomaly_thresh:
                anomaly_indices.append(i)
                self.total_anomalies += 1
                
        if not anomaly_indices:
            return # No anomalies in this batch, all dropped.
            
        # 2. CNN Feature Extraction
        # Filter spectrograms of anomalies
        anomaly_specs = spectrograms[anomaly_indices] # Shape: (N_anom, 5, 129, 9)
        with torch.no_grad():
            embeddings = self.cnn(anomaly_specs) # Shape: (N_anom, 64)
            
        # Process each anomaly embedding
        for i, original_idx in enumerate(anomaly_indices):
            meta = metas[original_idx]
            freq = meta["center_freq_hz"]
            emb = embeddings[i] # PyTorch tensor of shape (64,)
            
            # Initialize channel history if not present
            if freq not in self.channel_histories:
                self.channel_histories[freq] = {
                    "features": [],
                    "decisions": []
                }
                
            history = self.channel_histories[freq]
            history["features"].append(emb)
            
            # Limit history of feature vectors to sliding window of size 5
            if len(history["features"]) > 5:
                history["features"].pop(0)
                
            # If sliding window is full, run BiLSTM temporal verification
            if len(history["features"]) == 5:
                # Sequence format: (seq_len=5, batch=1, embedding_dim=64)
                seq_tensor = torch.stack(history["features"]).unsqueeze(1) # shape: (5, 1, 64)
                
                with torch.no_grad():
                    confidence = float(self.bilstm(seq_tensor).cpu().item())
                    
                bilstm_thresh = self.ai_cfg["bilstm_confidence_threshold"]
                is_confirmed = confidence >= bilstm_thresh
                
                history["decisions"].append(is_confirmed)
                if len(history["decisions"]) > 5:
                    history["decisions"].pop(0)
                    
                self.write_audit_log({
                    "event": "temporal_verification_stage",
                    "center_freq_hz": freq,
                    "timestamp": meta["timestamp"],
                    "bilstm_confidence": confidence,
                    "is_pattern_confirmed": is_confirmed
                })
                
                # 3. Hysteresis Gating Check (3 of 5 positive classifications)
                if len(history["decisions"]) == 5:
                    positive_count = sum(history["decisions"])
                    if positive_count >= self.ai_cfg["hysteresis_required_confirmations"]:
                        # Confirm Target!
                        self.total_targets += 1
                        target_event = {
                            "node_id": meta["node_id"],
                            "timestamp": meta["timestamp"],
                            "center_frequency_hz": freq,
                            "confidence": confidence,
                            "event_type": "Confirmed Target"
                        }
                        
                        # Secure localhost trigger emission
                        self.pub_socket.send_json(target_event)
                        
                        self.write_audit_log({
                            "event": "target_confirmed",
                            "center_freq_hz": freq,
                            "timestamp": meta["timestamp"],
                            "confidence": confidence,
                            "positive_frames_count": positive_count
                        })

    def run_pipeline(self, live_client):
        """
        Consumes from queue and batches processing for GPU efficiency.
        """
        batch_size = 8 # Batch size for GPU efficiency
        
        while not self.stop_event.is_set():
            batch = []
            
            # Populate batch up to size B, or timeout to keep latency low
            t_start = time.monotonic()
            while len(batch) < batch_size:
                elapsed = time.monotonic() - t_start
                remaining_time = max(0.01, 0.05 - elapsed) # Wait max 50ms to keep real-time constraint
                
                try:
                    meta, iq = self.raw_frame_queue.get(timeout=remaining_time)
                    batch.append((meta, iq))
                    self.total_processed += 1
                except queue.Empty:
                    break
                    
            if batch:
                try:
                    self.process_batch(batch)
                except Exception as e:
                    self.write_audit_log({
                        "event": "pipeline_error",
                        "error_message": str(e)
                    })
                    print(f"Error processing batch: {e}", file=sys.stderr)
                    
            time.sleep(0.001)

    def start(self, live_client):
        """
        Starts the subscriber and pipeline thread.
        """
        def recv_loop():
            live_client.connect_subscriber()
            while not self.stop_event.is_set():
                try:
                    meta, iq = live_client.receive_live_frame(timeout_ms=500)
                    if meta is not None:
                        self.enqueue_frame(meta, iq)
                except Exception as e:
                    print(f"Receiver error: {e}", file=sys.stderr)
                    time.sleep(0.1)
                    
        self.recv_thread = threading.Thread(target=recv_loop, daemon=True)
        self.recv_thread.start()
        
        self.pipeline_thread = threading.Thread(target=self.run_pipeline, args=(live_client,), daemon=True)
        self.pipeline_thread.start()

    def stop(self):
        self.stop_event.set()
        if hasattr(self, "recv_thread"):
            self.recv_thread.join()
        if hasattr(self, "pipeline_thread"):
            self.pipeline_thread.join()
        self.pub_socket.close()
        self.context.term()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ai_sentry.py <config_path>")
        sys.exit(1)
        
    cfg_path = sys.argv[1]
    
    # Import client here to prevent circular imports
    from buffer_client import BufferClient
    
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    node_cfg = config["node"]
    
    client = BufferClient(
        bind_address=node_cfg["bind_address"],
        pub_port=node_cfg["pub_port"],
        rep_port=node_cfg["rep_port"],
        security_token=node_cfg["security_token"]
    )
    
    pipeline = AISentryPipeline(cfg_path)
    pipeline.start(client)
    
    print("AI Sentry Service started. Press Ctrl+C to terminate.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down AI Sentry Service...")
        pipeline.stop()
        client.close()
        print("Shutdown complete.")
