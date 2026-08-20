import os
import sys
import time
import yaml
import json
import zmq
import torch
import numpy as np
import threading
from buffer_client import BufferClient

def precompute_steering_vectors(r, c, f, num_antennas, grid_step_deg, device):
    """
    Precomputes steering vectors for UCA on the search grid (Azimuth [0, 360], Elevation [0, 90]).
    Returns a unified steering matrix of shape (GridSize, num_antennas).
    """
    lam = c / f
    azimuths = torch.arange(0.0, 360.0, grid_step_deg, device=device)
    elevations = torch.arange(0.0, 90.01, grid_step_deg, device=device)
    
    az_grid, el_grid = torch.meshgrid(azimuths, elevations, indexing='ij')
    az_flat = torch.deg2rad(az_grid.flatten())
    el_flat = torch.deg2rad(el_grid.flatten())
    
    m_idx = torch.arange(num_antennas, device=device)
    gamma = m_idx * 2 * np.pi / num_antennas
    
    # Steering vector phase: (2 * pi * r / lambda) * cos(phi) * cos(theta - gamma)
    phase_coef = (2 * np.pi * r) / lam
    theta_gamma_diff = az_flat.unsqueeze(1) - gamma.unsqueeze(0)
    cos_diff = torch.cos(theta_gamma_diff)
    cos_el = torch.cos(el_flat).unsqueeze(1)
    
    phases = phase_coef * cos_el * cos_diff
    steering_matrix = torch.exp(1j * phases)
    
    return steering_matrix, azimuths, elevations

def apply_fbss(R):
    """
    Applies Forward/Backward Covariance Averaging to decorrelate coherent multipath.
    Valid for any arbitrary geometry (including UCA).
    """
    M = R.shape[0]
    J = torch.eye(M, device=R.device, dtype=R.dtype).flip(1)
    R_fb = 0.5 * (R + J @ torch.conj(R) @ J)
    return R_fb

def regularize_covariance(R, alpha=1e-6):
    """
    Regularizes covariance matrix using diagonal loading.
    Prevents singular matrix errors during Eigen-decomposition.
    """
    M = R.shape[0]
    trace = torch.trace(R).real
    loading = alpha * trace
    R_reg = R + loading * torch.eye(M, device=R.device, dtype=R.dtype)
    return R_reg

def run_music(iq_matrix, steering_matrix, azimuths, elevations, snr_threshold_db, device):
    """
    Executes MUSIC algorithm using regularized covariance and precomputed steering vectors.
    """
    # iq_matrix shape: (M, N) complex64
    M, N = iq_matrix.shape
    t_iq = torch.from_numpy(iq_matrix).to(device)
    
    # Compute spatial covariance R = (1/N) * X * X^H
    R = (t_iq @ torch.conj(t_iq.t())) / N
    
    # Apply Forward/Backward Covariance Averaging
    R_fb = apply_fbss(R)
    
    # Regularize
    R_reg = regularize_covariance(R_fb)
    
    # Eigen-decomposition
    try:
        evals, evecs = torch.linalg.eigh(R_reg)
    except RuntimeError as e:
        print(f"Eigen-decomposition failed: {e}", file=sys.stderr)
        return []
        
    # Noise Subspace: first M - D eigenvectors (assume D=1 dominant target)
    En = evecs[:, :M-1] # Shape: (5, 4)
    
    # Project steering matrix: den = sum(|a^H * En|^2)
    # Shape of projection: (GridSize, M) @ (M, 4) -> (GridSize, 4)
    projected = steering_matrix @ En
    den = torch.sum(torch.abs(projected) ** 2, dim=1)
    
    # MUSIC Power spectrum
    P = 1.0 / torch.clamp(den, min=1e-12)
    
    # Normalize and convert to dB
    P_min = torch.min(P)
    P_db = 10 * torch.log10(P / P_min)
    
    # Reshape to grid
    num_az = len(azimuths)
    num_el = len(elevations)
    P_grid = P_db.view(num_az, num_el)
    
    # Find Peaks (Local Maxima)
    peaks = []
    for i in range(num_az):
        for j in range(num_el):
            val = P_grid[i, j].item()
            if val < snr_threshold_db:
                continue
                
            # Check 8-neighbors (Azimuth is circular, Elevation is bounded)
            i_prev = (i - 1) % num_az
            i_next = (i + 1) % num_az
            j_prev = max(0, j - 1)
            j_next = min(num_el - 1, j + 1)
            
            is_max = True
            for ni in (i_prev, i, i_next):
                for nj in (j_prev, j, j_next):
                    if ni == i and nj == j:
                        continue
                    if P_grid[ni, nj].item() > val:
                        is_max = False
                        break
                if not is_max:
                    break
                    
            if is_max:
                peaks.append({
                    "azimuth": float(azimuths[i].item()),
                    "elevation": float(elevations[j].item()),
                    "snr": val
                })
                
    return peaks

class GeometryEngine:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.node_cfg = self.config["node"]
        self.sdr_cfg = self.config["sdr"]
        self.geo_cfg = self.config["geometry_engine"]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Geometry Engine running on device: {self.device}")
        
        # Precompute steering vectors for 2.4 GHz and 5.8 GHz bands
        # Precomputations cache to ensure sub-millisecond execution times
        self.steering_cache = {}
        for freq in self.sdr_cfg["center_freqs_hz"]:
            sm, az, el = precompute_steering_vectors(
                r=self.geo_cfg["array_radius_m"],
                c=self.geo_cfg["speed_of_light_mps"],
                f=freq,
                num_antennas=self.sdr_cfg["num_antennas"],
                grid_step_deg=self.geo_cfg["grid_step_deg"],
                device=self.device
            )
            self.steering_cache[freq] = (sm, az, el)
            
        # ZMQ Context
        self.context = zmq.Context()
        
        # Subscribe to Confirmed Target queue from Phase 2 Sentry
        self.sub_target = self.context.socket(zmq.SUB)
        self.sub_target.setsockopt(zmq.SUBSCRIBE, b"")
        self.sub_target.connect(f"tcp://127.0.0.1:{self.config['ai_sentry']['trigger_pub_port']}")
        
        # Publisher for AoA calculations
        self.pub_angles = self.context.socket(zmq.PUB)
        self.pub_angles.setsockopt(zmq.SNDHWM, 1000)
        self.pub_angles.bind(f"tcp://{self.node_cfg['bind_address']}:{self.geo_cfg['angles_pub_port']}")
        
        self.stop_event = threading.Event()
        self.client = None

    def get_buffer_client(self):
        if self.client is None:
            self.client = BufferClient(
                bind_address=self.node_cfg["bind_address"],
                pub_port=self.node_cfg["pub_port"],
                rep_port=self.node_cfg["rep_port"],
                security_token=self.node_cfg["security_token"]
            )
        return self.client

    def on_confirmed_target(self, event):
        """
        Calculates Azimuth and Elevation upon target confirmation.
        """
        timestamp = event["timestamp"]
        freq = event["center_frequency_hz"]
        
        # 1. Reach back into the Phase 1 buffer client
        client = self.get_buffer_client()
        
        # Query exactly around the trigger timestamp
        # Re-verify token security on client connection
        try:
            meta, data = client.query_history(
                start_time=timestamp - 0.005,
                # Give a small 10ms window to find the frame
                end_time=timestamp + 0.005,
                frequency_hz=freq,
                timeout_ms=1000
            )
        except Exception as e:
            print(f"Error querying buffer: {e}", file=sys.stderr)
            return
            
        if meta["match_count"] == 0:
            print(f"No matching historical data found for target at {timestamp}s", file=sys.stderr)
            return
            
        # 2. Strict shape validation
        # Shape of single frame is (5, 1024). Batch shape from query is (matches, 5, 1024).
        # We process the first matched frame.
        iq_frame = data[0]
        expected_shape = (self.sdr_cfg["num_antennas"], self.sdr_cfg["frame_size_samples"])
        if iq_frame.shape != expected_shape:
            # Drop frame immediately to mitigate memory corruption / buffer overflow vulnerabilities
            print(f"SECURITY EXCEPTION: Malformed frame shape received: {iq_frame.shape}. Expected: {expected_shape}", file=sys.stderr)
            return
            
        # Get cache
        if freq not in self.steering_cache:
            print(f"Error: Steering vector cache missing for frequency: {freq}", file=sys.stderr)
            return
            
        sm, az, el = self.steering_cache[freq]
        
        # 3. Run MUSIC estimation on GPU
        peaks = run_music(
            iq_matrix=iq_frame,
            steering_matrix=sm,
            azimuths=az,
            elevations=el,
            snr_threshold_db=self.geo_cfg["snr_threshold_db"],
            device=self.device
        )
        
        # 4. Emit results to Phase 4
        if peaks:
            angles_event = {
                "node_id": event["node_id"],
                "timestamp": timestamp,
                "center_frequency_hz": freq,
                "peaks": peaks
            }
            self.pub_angles.send_json(angles_event)
            
        # GPU Memory clean up between calls
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def run(self):
        print("Geometry Engine running and listening for triggers...")
        poller = zmq.Poller()
        poller.register(self.sub_target, zmq.POLLIN)
        
        while not self.stop_event.is_set():
            socks = dict(poller.poll(timeout=100))
            if self.sub_target in socks:
                try:
                    event = self.sub_target.recv_json()
                    self.on_confirmed_target(event)
                except Exception as e:
                    print(f"Error in confirmed target trigger loop: {e}", file=sys.stderr)
                    
        self.sub_target.close()
        self.pub_angles.close()
        if self.client:
            self.client.close()
        self.context.term()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python geometry.py <config_path>")
        sys.exit(1)
        
    cfg_path = sys.argv[1]
    engine = GeometryEngine(cfg_path)
    
    try:
        engine.run()
    except KeyboardInterrupt:
        print("Shutting down Geometry Engine...")
        engine.stop_event.set()
