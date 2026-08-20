import os
import sys
import time
import yaml
import zmq
import numpy as np
import scipy.io as sio

def find_iq_variable(mat_dict):
    """
    Scans the keys of the MATLAB dictionary to find the complex/floating-point signal variable.
    """
    for key, val in mat_dict.items():
        if key.startswith('__'):
            continue  # Skip MATLAB metadata keys
        if isinstance(val, np.ndarray):
            # Check if it is a numeric array containing float or complex numbers
            if np.issubdtype(val.dtype, np.number):
                return val
    return None

def play_mat_dataset(config_path, dataset_folder):
    # 1. Load system config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    node_cfg = config["node"]
    sdr_cfg = config["sdr"]
    
    # 2. Setup ZMQ Publisher
    context = zmq.Context()
    pub_socket = context.socket(zmq.PUB)
    pub_socket.setsockopt(zmq.SNDHWM, 1000)
    pub_socket.bind(f"tcp://{node_cfg['bind_address']}:{node_cfg['pub_port']}")
    
    # 3. Stream parameters
    frame_size = sdr_cfg["frame_size_samples"]
    num_antennas = sdr_cfg["num_antennas"]
    frame_duration = frame_size / sdr_cfg["sample_rate_hz"]
    
    print(f"Scanning for .mat files in: {dataset_folder}")
    
    # We use a generator (os.walk) instead of listing all files in memory at once.
    # This keeps memory usage extremely low even with 100,000+ files.
    file_count = 0
    
    while True:
        has_files = False
        for root, dirs, files in os.walk(dataset_folder):
            for file in files:
                if not file.endswith(".mat"):
                    continue
                
                has_files = True
                filepath = os.path.join(root, file)
                file_count += 1
                
                if file_count % 500 == 0:
                    print(f"Processed {file_count} files...")
                
                try:
                    # Load the MATLAB file
                    mat_data = sio.loadmat(filepath)
                    data = find_iq_variable(mat_data)
                    
                    if data is None:
                        print(f"Warning: No valid numeric array found in {file}. Skipping.")
                        continue
                    
                    # Convert to complex64 if not already
                    data = data.astype(np.complex64)
                    
                    # Check shape and transpose if necessary: 
                    # We expect shape (num_antennas, length) e.g., (5, N)
                    if len(data.shape) == 1:
                        # 1D array: duplicate across all antennas
                        data = np.tile(data, (num_antennas, 1))
                    elif data.shape[0] != num_antennas and data.shape[1] == num_antennas:
                        # Shape is (N, 5): transpose to (5, N)
                        data = data.T
                    elif data.shape[0] != num_antennas:
                        # Fallback: if dimension mismatch, tile or pad
                        data = np.tile(data[0, :], (num_antennas, 1))
                        
                    length = data.shape[1]
                    if length < frame_size:
                        # Frame is too short, pad with zeros
                        padding = frame_size - length
                        data = np.pad(data, ((0,0), (0, padding)), mode='constant')
                        length = frame_size
                    
                    # Stream chunks of size frame_size
                    for start in range(0, length - frame_size + 1, frame_size):
                        t_start = time.monotonic()
                        iq_frame = data[:, start:start+frame_size]
                        
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
                    print(f"Error reading or streaming file {file}: {e}")
                    
        if not has_files:
            print("No .mat files found in the specified directory.")
            break
            
        print("Completed one full run over the dataset. Restarting stream loop...")
        time.sleep(2.0)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python play_mat_dataset.py <config_path> <dataset_folder_path>")
        sys.exit(1)
    play_mat_dataset(sys.argv[1], sys.argv[2])
