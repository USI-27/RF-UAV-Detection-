import os
import sys
import time
import yaml
import zmq
import numpy as np

def play_dataset(config_path, dataset_folder):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    node_cfg = config["node"]
    sdr_cfg = config["sdr"]
    
    # Setup ZMQ Publisher
    context = zmq.Context()
    pub_socket = context.socket(zmq.PUB)
    pub_socket.setsockopt(zmq.SNDHWM, 1000)
    pub_socket.bind(f"tcp://{node_cfg['bind_address']}:{node_cfg['pub_port']}")
    
    # Retrieve files from your dataset folder
    files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith(".npy")]
    if not files:
        print(f"No .npy dataset files found in {dataset_folder}")
        return
        
    print(f"Dataset player started. Streaming {len(files)} files on port {node_cfg['pub_port']}...")
    
    frame_size = sdr_cfg["frame_size_samples"]
    num_antennas = sdr_cfg["num_antennas"]
    frame_duration = frame_size / sdr_cfg["sample_rate_hz"]
    
    file_idx = 0
    while True:
        # Load file data
        filepath = files[file_idx]
        print(f"Playing file: {os.path.basename(filepath)}")
        
        try:
            # Load complex I/Q matrix. Shape should ideally be (num_antennas, length)
            # e.g., shape (5, N)
            data = np.load(filepath) 
            
            # If shape is 1D, replicate across antennas for testing
            if len(data.shape) == 1:
                data = np.tile(data, (num_antennas, 1))
                
            length = data.shape[1]
            
            # Stream chunks of size frame_size
            for start in range(0, length - frame_size, frame_size):
                t_start = time.monotonic()
                iq_frame = data[:, start:start+frame_size].astype(np.complex64)
                
                # Send frame metadata
                metadata = {
                    "node_id": node_cfg["id"],
                    "timestamp": time.time(),
                    "clock_synced": True,
                    "center_freq_hz": sdr_cfg["center_freqs_hz"][0],
                    "sample_rate_hz": sdr_cfg["sample_rate_hz"],
                    "shape": list(iq_frame.shape),
                    "dtype": str(iq_frame.dtype),
                    "token": node_cfg["security_token"]
                }
                
                pub_socket.send_json(metadata, zmq.SNDMORE)
                pub_socket.send(iq_frame.tobytes())
                
                # Match real-time sample rate pacing
                elapsed = time.monotonic() - t_start
                sleep_time = frame_duration - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except Exception as e:
            print(f"Error reading or streaming file {filepath}: {e}")
                
        file_idx = (file_idx + 1) % len(files)
        time.sleep(1.0) # Pause between files

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python play_dataset.py <config_path> <dataset_folder_path>")
        sys.exit(1)
    play_dataset(sys.argv[1], sys.argv[2])
