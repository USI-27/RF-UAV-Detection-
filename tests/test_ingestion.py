import os
import sys
import time
import subprocess
import unittest
import urllib.request
import numpy as np
import yaml
import socket

# Add root folder to python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from buffer_client import BufferClient

class TestIngestionAndBuffer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = "config.yaml"
        cls.cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Load config details
        with open(cls.config_path, "r") as f:
            cls.config = yaml.safe_load(f)
            
        cls.node_cfg = cls.config["node"]
        cls.sdr_cfg = cls.config["sdr"]
        
        # Start ingestion service as a subprocess
        print("Starting ingestion.py subprocess...")
        cls.process = subprocess.Popen(
            [sys.executable, "ingestion.py", cls.config_path],
            cwd=cls.cwd
        )
        # Give processes some time to bind sockets and initialize
        time.sleep(2)

    @classmethod
    def tearDownClass(cls):
        # Gracefully terminate the process
        print("Terminating ingestion.py subprocess...")
        cls.process.terminate()
        try:
            cls.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.process.kill()

    def test_localhost_binding_only(self):
        """
        Security Verification: Verify that ZMQ port is only listening on localhost
        and not exposing 0.0.0.0.
        """
        # Get the external IP address of the machine
        try:
            hostname = socket.gethostname()
            external_ip = socket.gethostbyname(hostname)
        except Exception:
            external_ip = "127.0.0.1"
            
        # Try to connect to the pub port on external IP; it should fail/refuse connection
        if external_ip != "127.0.0.1":
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0)
            result = s.connect_ex((external_ip, self.node_cfg["pub_port"]))
            self.assertNotEqual(result, 0, f"ZMQ pub port exposed on external interface {external_ip}!")
            s.close()

    def test_live_stream_and_data_integrity(self):
        """
        Data Integrity: Subscribe to live feed and check frame shapes, types and coherence.
        """
        client = BufferClient(
            bind_address=self.node_cfg["bind_address"],
            pub_port=self.node_cfg["pub_port"],
            rep_port=self.node_cfg["rep_port"],
            security_token=self.node_cfg["security_token"]
        )
        client.connect_subscriber()
        
        # Receive 3 consecutive frames
        for _ in range(3):
            meta, iq_data = client.receive_live_frame(timeout_ms=2000)
            self.assertIsNotNone(meta, "Failed to receive live frame within timeout")
            self.assertEqual(meta["node_id"], self.node_cfg["id"])
            self.assertTrue(meta["clock_synced"])
            
            # Validate shape matches M x N array representation
            expected_shape = (self.sdr_cfg["num_antennas"], self.sdr_cfg["frame_size_samples"])
            self.assertEqual(iq_data.shape, expected_shape)
            self.assertEqual(iq_data.dtype, np.complex64)
            
            # Verify data is non-zero (checking coherent generator status)
            self.assertTrue(np.any(iq_data))
            
        client.close()

    def test_security_auth_handling(self):
        """
        Security Verification: Verify that unauthorized requests (invalid security token) are blocked.
        """
        client_bad = BufferClient(
            bind_address=self.node_cfg["bind_address"],
            pub_port=self.node_cfg["pub_port"],
            rep_port=self.node_cfg["rep_port"],
            security_token="WRONG_TOKEN"
        )
        
        # Query with bad token should fail or receive empty/error message
        t_now = time.time()
        with self.assertRaises(Exception):
            client_bad.query_history(start_time=t_now - 10, end_time=t_now, timeout_ms=2000)
            
        client_bad.close()

    def test_historical_random_access_replay(self):
        """
        Random-Access Query: Retrieve historical buffer slices by timestamp.
        """
        client = BufferClient(
            bind_address=self.node_cfg["bind_address"],
            pub_port=self.node_cfg["pub_port"],
            rep_port=self.node_cfg["rep_port"],
            security_token=self.node_cfg["security_token"]
        )
        
        # Capture current time
        t_start = time.time()
        time.sleep(2) # Let some frames accumulate
        t_end = time.time()
        
        # Query history
        meta, history_data = client.query_history(start_time=t_start, end_time=t_end)
        self.assertIsNotNone(meta)
        self.assertIn("match_count", meta)
        
        if meta["match_count"] > 0:
            self.assertEqual(history_data.ndim, 3) # (matches, antennas, snapshots)
            self.assertEqual(history_data.shape[0], meta["match_count"])
            self.assertEqual(history_data.shape[1], self.sdr_cfg["num_antennas"])
            self.assertEqual(history_data.shape[2], self.sdr_cfg["frame_size_samples"])
            
        client.close()

    def test_metrics_http_endpoint(self):
        """
        Health and Diagnostic verification: Pull metrics from HTTP endpoint.
        """
        url = f"http://{self.node_cfg['bind_address']}:{self.node_cfg['metrics_port']}/metrics"
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                html = response.read().decode("utf-8")
                
                # Check for standard Prometheus labels
                self.assertIn("sdr_samples_received_total", html)
                self.assertIn("sdr_dropped_frames_total", html)
                self.assertIn("sdr_buffer_occupancy_ratio", html)
                self.assertIn("clock_skew_seconds", html)
        except Exception as e:
            self.fail(f"Failed to access Prometheus metrics server: {e}")

if __name__ == "__main__":
    unittest.main()
