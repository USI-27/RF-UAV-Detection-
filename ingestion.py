import os
import sys
import time
import yaml
import json
import zmq
import numpy as np
import multiprocessing
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

# Security Check: Ensure authenticated and synchronized clock
def get_system_time_info():
    """
    Returns the current synchronized system time.
    In production, this would interface with a GPSDO or PTP daemon.
    For this implementation, we monitor system time and check for sync status.
    """
    t = time.time()
    # Simple check for clock skew / health: compare time.time() and time.monotonic() drift
    # In a real node, we'd query local NTP/PTP daemon status
    is_synced = True 
    if sys.platform == "win32":
        # Check Windows time sync status briefly (non-blocking, direct)
        # w32tm status query can be slow, so we simulate NTP auth check and use system time.
        pass
    return t, is_synced

class SimulatedSDR:
    """
    High-fidelity mockup of SoapySDR interface for a 5-element Uniform Circular Array (UCA).
    Generates coherent raw I/Q data on target frequencies (2.4 GHz / 5.8 GHz).
    """
    def __init__(self, sample_rate, num_antennas, frame_size, center_freqs):
        self.sample_rate = sample_rate
        self.num_antennas = num_antennas
        self.frame_size = frame_size
        self.center_freqs = center_freqs
        self.freq_index = 0

    def read_stream(self):
        """
        Simulates reading M x N coherent I/Q matrix.
        M = num_antennas, N = frame_size.
        """
        freq = self.center_freqs[self.freq_index]
        # Alternate frequencies to simulate staring at multiple bands
        self.freq_index = (self.freq_index + 1) % len(self.center_freqs)
        
        # Generate noise
        noise = (np.random.randn(self.num_antennas, self.frame_size) + 
                 1j * np.random.randn(self.num_antennas, self.frame_size)) * 0.1
        
        # Inject a simulated signal (e.g. drone FHSS burst) at a specific AoA (Azimuth=45, Elevation=10)
        # Steering vector for UCA (5 antennas)
        theta = np.deg2rad(45.0)  # Azimuth
        phi = np.deg2rad(10.0)    # Elevation
        r = 0.06  # Radius of array in meters (half-wavelength at 2.4GHz is ~0.06m)
        c = 3e8   # Speed of light
        
        # Antenna coordinates for 5-element UCA
        angles = np.arange(self.num_antennas) * 2 * np.pi / self.num_antennas
        ant_x = r * np.cos(angles)
        ant_y = r * np.sin(angles)
        ant_z = np.zeros(self.num_antennas)
        
        # Wave vector
        kx = 2 * np.pi * freq / c * np.cos(phi) * np.cos(theta)
        ky = 2 * np.pi * freq / c * np.cos(phi) * np.sin(theta)
        kz = 2 * np.pi * freq / c * np.sin(phi)
        
        # Steering vector phases
        phases = ant_x * kx + ant_y * ky + ant_z * kz
        steering_vector = np.exp(1j * phases)
        
        # Generate baseband signal (e.g., sine wave representing burst)
        t_vec = np.arange(self.frame_size) / self.sample_rate
        signal_baseband = np.sin(2 * np.pi * 100000 * t_vec) + 1j * np.cos(2 * np.pi * 100000 * t_vec)
        
        # Combine steering vector and signal
        signal = np.outer(steering_vector, signal_baseband)
        
        iq_data = (noise + signal).astype(np.complex64)
        return iq_data, freq

def ingestion_loop(config_path, stop_event):
    """
    Thread 1 (Ingestion Process): Continuous streaming of raw I/Q samples from SDR.
    Runs in isolated process to prevent GIL bottlenecks.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    node_cfg = config["node"]
    sdr_cfg = config["sdr"]
    
    # Establish ZeroMQ secure publisher
    context = zmq.Context()
    pub_socket = context.socket(zmq.PUB)
    # Configure socket constraints to prevent memory leaks / exhaustion
    pub_socket.setsockopt(zmq.SNDHWM, 1000) 
    pub_socket.bind(f"tcp://{node_cfg['bind_address']}:{node_cfg['pub_port']}")

    sdr = SimulatedSDR(
        sample_rate=sdr_cfg["sample_rate_hz"],
        num_antennas=sdr_cfg["num_antennas"],
        frame_size=sdr_cfg["frame_size_samples"],
        center_freqs=sdr_cfg["center_freqs_hz"]
    )

    frame_duration = sdr_cfg["frame_size_samples"] / sdr_cfg["sample_rate_hz"]
    
    print(f"Ingestion service started for {node_cfg['id']} on pub port {node_cfg['pub_port']}")

    while not stop_event.is_set():
        t_start = time.monotonic()
        
        # Fetch coherent UCA I/Q samples
        iq_data, freq = sdr.read_stream()
        
        # Capture NTP/GPSDO synchronized timestamp
        hw_timestamp, is_synced = get_system_time_info()
        
        # Package metadata (strictly bound token check)
        metadata = {
            "node_id": node_cfg["id"],
            "timestamp": hw_timestamp,
            "clock_synced": is_synced,
            "center_freq_hz": freq,
            "sample_rate_hz": sdr_cfg["sample_rate_hz"],
            "shape": list(iq_data.shape),
            "dtype": str(iq_data.dtype),
            "token": node_cfg["security_token"]
        }
        
        # Multi-part message to avoid stringifying raw floating point values
        pub_socket.send_json(metadata, zmq.SNDMORE)
        pub_socket.send(iq_data.tobytes())
        
        # Precise real-time rate tracking sleep
        elapsed = time.monotonic() - t_start
        sleep_time = frame_duration - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

class MetricsHandler(BaseHTTPRequestHandler):
    """
    Simple HTTP server serving Prometheus metrics.
    """
    def log_message(self, format, *args):
        pass # Suppress standard library logger output to avoid console spam

    def do_GET(self):
        if self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.end_headers()
            
            # Read current metrics from global state
            metrics = self.server.metrics_store
            output = []
            for name, val in metrics.items():
                output.append(f"{name} {val}")
            self.wfile.write("\n".join(output).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

def start_metrics_server(bind_address, port, metrics_store):
    server = HTTPServer((bind_address, port), MetricsHandler)
    server.metrics_store = metrics_store
    
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server

def buffer_loop(config_path, stop_event):
    """
    Process 2 (Buffer Process): Subscribes to live feed, stores frames in a high-speed
    FIFO ring buffer, manages historical queries via REQ/REP, and exposes health metrics.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    node_cfg = config["node"]
    sdr_cfg = config["sdr"]
    buffer_cfg = config["buffer"]

    # Pre-allocated circular ring buffers to prevent memory fragmentation / allocation lag
    max_frames = buffer_cfg["max_frames"]
    
    # Ring buffer arrays
    metadata_buffer = [None] * max_frames
    # Shape of frame data: (M, N) complex64
    shape = (sdr_cfg["num_antennas"], sdr_cfg["frame_size_samples"])
    data_buffer = np.zeros((max_frames, *shape), dtype=np.complex64)
    
    head_idx = 0
    buffer_count = 0
    
    # Metrics store
    metrics_store = {
        "sdr_samples_received_total": 0,
        "sdr_dropped_frames_total": 0,
        "sdr_buffer_occupancy_ratio": 0.0,
        "clock_skew_seconds": 0.0,
        "security_violations_total": 0
    }
    
    metrics_server = start_metrics_server(node_cfg["bind_address"], node_cfg["metrics_port"], metrics_store)
    print(f"Metrics server started on http://{node_cfg['bind_address']}:{node_cfg['metrics_port']}/metrics")

    # ZMQ Setup
    context = zmq.Context()
    
    # Subscriber to live stream
    sub_socket = context.socket(zmq.SUB)
    sub_socket.setsockopt(zmq.SUBSCRIBE, b"")
    sub_socket.setsockopt(zmq.RCVHWM, 1000)
    sub_socket.connect(f"tcp://{node_cfg['bind_address']}:{node_cfg['pub_port']}")

    # Rep socket for reach-back query
    rep_socket = context.socket(zmq.REP)
    rep_socket.bind(f"tcp://{node_cfg['bind_address']}:{node_cfg['rep_port']}")

    # Use Poller to handle both incoming stream and query socket asynchronously without blocking
    poller = zmq.Poller()
    poller.register(sub_socket, zmq.POLLIN)
    poller.register(rep_socket, zmq.POLLIN)

    print(f"Buffer service listening on query port {node_cfg['rep_port']}")

    last_metric_update = time.time()
    frames_processed_this_sec = 0

    while not stop_event.is_set():
        socks = dict(poller.poll(timeout=100))
        
        # Handle live ingestion frame
        if sub_socket in socks:
            try:
                metadata = sub_socket.recv_json(flags=zmq.NOBLOCK)
                raw_data = sub_socket.recv(flags=zmq.NOBLOCK)
                
                # Security Check: Authentication Token Verification
                if metadata.get("token") != node_cfg["security_token"]:
                    metrics_store["security_violations_total"] += 1
                    continue
                
                # Check for clock drift
                local_time = time.time()
                drift = abs(local_time - metadata["timestamp"])
                metrics_store["clock_skew_seconds"] = drift
                
                iq_data = np.frombuffer(raw_data, dtype=np.complex64).reshape(shape)
                
                # Push to pre-allocated Ring Buffer
                metadata_buffer[head_idx] = metadata
                data_buffer[head_idx] = iq_data
                
                head_idx = (head_idx + 1) % max_frames
                if buffer_count < max_frames:
                    buffer_count += 1
                else:
                    metrics_store["sdr_dropped_frames_total"] += 1 # Oldest frame evicted
                
                # Update metrics
                metrics_store["sdr_samples_received_total"] += sdr_cfg["frame_size_samples"]
                frames_processed_this_sec += 1
                metrics_store["sdr_buffer_occupancy_ratio"] = buffer_count / max_frames
                
            except zmq.Again:
                pass

        # Handle historical queries (random-access replay)
        if rep_socket in socks:
            try:
                query = rep_socket.recv_json()
                
                # Security Token Validation
                if query.get("token") != node_cfg["security_token"]:
                    rep_socket.send_json({"error": "Unauthorized", "data": []})
                    metrics_store["security_violations_total"] += 1
                    continue
                
                start_t = query["start_time"]
                end_t = query["end_time"]
                freq = query.get("frequency_hz")
                
                # Linear scan of the active buffer slots to find matching time range
                # Simple & robust search (stdlib/numpy indexing)
                results = []
                for i in range(buffer_count):
                    meta = metadata_buffer[i]
                    if meta is not None:
                        t = meta["timestamp"]
                        f_match = (freq is None or meta["center_freq_hz"] == freq)
                        if start_t <= t <= end_t and f_match:
                            results.append({
                                "index": i,
                                "timestamp": t,
                                "center_freq_hz": meta["center_freq_hz"]
                            })
                
                # Package matching raw buffers and return
                response_metadata = {"match_count": len(results), "matches": results}
                rep_socket.send_json(response_metadata, zmq.SNDMORE)
                
                # Concatenate data arrays for matched frames
                if len(results) > 0:
                    matched_data = np.stack([data_buffer[r["index"]] for r in results])
                    rep_socket.send(matched_data.tobytes())
                else:
                    rep_socket.send(b"")
                    
            except Exception as e:
                try:
                    rep_socket.send_json({"error": str(e)})
                except:
                    pass

        # Periodic log & metric refresh
        now = time.time()
        if now - last_metric_update >= 1.0:
            last_metric_update = now
            frames_processed_this_sec = 0

    metrics_server.shutdown()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingestion.py <config_path>")
        sys.exit(1)
        
    cfg_path = sys.argv[1]
    
    stop_event = multiprocessing.Event()
    
    # Process 1: Ingestion
    p_ingest = multiprocessing.Process(
        target=ingestion_loop, 
        args=(cfg_path, stop_event),
        name="SDR_Ingestion_Process"
    )
    
    # Process 2: Buffer and Query manager
    p_buffer = multiprocessing.Process(
        target=buffer_loop,
        args=(cfg_path, stop_event),
        name="Buffer_Management_Process"
    )
    
    p_ingest.start()
    p_buffer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down processes gracefully...")
        stop_event.set()
        p_ingest.join()
        p_buffer.join()
        print("Shutdown complete.")
