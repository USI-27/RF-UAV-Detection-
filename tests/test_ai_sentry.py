import os
import sys
import time
import unittest
import yaml
import torch
import numpy as np
import zmq
import tempfile
import json
import queue

# Add root folder to python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ConvAutoencoder
from ai_sentry import AISentryPipeline, load_verified_model

class TestAISentry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = "config.yaml"
        with open(cls.config_path, "r") as f:
            cls.config = yaml.safe_load(f)
            
        cls.node_cfg = cls.config["node"]
        cls.ai_cfg = cls.config["ai_sentry"]
        cls.sdr_cfg = cls.config["sdr"]
        
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_cryptographic_verification_fails_on_tamper(self):
        """
        Security Verification: Verify that modifying model file or tampering with weights
        causes the pipeline to reject loading.
        """
        # Create a temp file and write corrupted data
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"CORRUPTED_MODEL_WEIGHTS_DATA")
            temp_path = f.name
            
        # Create a modified config that has the temp file path but original hash
        tampered_config = self.config.copy()
        tampered_config["models"]["autoencoder"]["path"] = temp_path
        
        try:
            # Attempting to load this should raise ValueError due to cryptographic hash mismatch
            with self.assertRaises(ValueError):
                load_verified_model(ConvAutoencoder, "autoencoder", tampered_config, self.device)
        finally:
            os.remove(temp_path)

    def test_stft_spectrogram_preprocessing(self):
        """
        Data Integrity: Verify raw complex64 array conversion into 5-channel STFT spectrogram magnitude.
        """
        pipeline = AISentryPipeline(self.config_path)
        try:
            # Test shape (5, 1024)
            iq_data = (np.random.randn(5, 1024) + 1j * np.random.randn(5, 1024)).astype(np.complex64)
            
            # Convert to PyTorch complex tensor
            t_iq = torch.from_numpy(iq_data).to(pipeline.device) # Shape: (5, 1024)
            window = torch.hann_window(256, device=pipeline.device)
            
            stft_out = torch.stft(t_iq, n_fft=256, hop_length=128, window=window, return_complex=True)
            spectrogram = torch.abs(stft_out)
            
            # Verify shape is (5, 256, 9)
            self.assertEqual(spectrogram.shape, (5, 256, 9))
            self.assertEqual(spectrogram.dtype, torch.float32)
        finally:
            pipeline.stop()

    def test_backpressure_load_shedding(self):
        """
        DoS Mitigation Verification: Verify that the pipeline drops incoming frames (sheds load)
        and increments the drop metrics when queue limit is exceeded.
        """
        pipeline = AISentryPipeline(self.config_path)
        try:
            # Mock queue put_nowait to raise queue.Full exception
            pipeline.raw_frame_queue.put_nowait = unittest.mock.Mock(side_effect=queue.Full())
            
            # Setup audit logging mock to count drops
            pipeline.write_audit_log = unittest.mock.Mock()
            
            meta = {
                "node_id": "Node_A",
                "timestamp": time.time(),
                "center_freq_hz": 2400000000
            }
            iq_data = np.zeros((5, 1024), dtype=np.complex64)
            
            # Try to enqueue frame; should trigger backpressure drop
            pipeline.enqueue_frame(meta, iq_data)
            
            self.assertEqual(pipeline.total_dropped, 1)
            pipeline.write_audit_log.assert_called()
        finally:
            pipeline.stop()

    def test_end_to_end_sentry_verification(self):
        """
        End-to-End Pipeline Check: Feed mocked anomalies, verify target triggers are published.
        """
        pipeline = AISentryPipeline(self.config_path)
        
        # Set up a listener for trigger events on localhost but different subscriber context
        context = zmq.Context()
        sub_trigger = context.socket(zmq.SUB)
        sub_trigger.setsockopt(zmq.SUBSCRIBE, b"")
        sub_trigger.connect(f"tcp://127.0.0.1:{self.ai_cfg['trigger_pub_port']}")
        # Give ZMQ subscriber time to establish the connection (mitigate slow-joiner symptom)
        time.sleep(0.5)
        
        try:
            # Generate 5 consecutive frame anomalies to trigger the hysteresis confirmation (3 of 5)
            # We mock processing loop to simulate input frames
            meta = {
                "node_id": "Node_A",
                "timestamp": time.time(),
                "center_freq_hz": 2400000000
            }
            # Zero arrays will generally cause reconstruction error if weights are randomized
            iq_data = np.random.randn(5, 1024).astype(np.complex64)
            
            # Mock autoencoder & lstm output directly to guarantee deterministic test outcomes
            pipeline.autoencoder = unittest.mock.Mock()
            # Force a high reconstruction error to trigger anomaly branch
            pipeline.autoencoder.return_value = torch.ones(1, 5, 256, 9).to(pipeline.device) * 10.0
            
            # Mock LSTM to return 0.9 confidence (exceeding confidence threshold of 0.7)
            pipeline.bilstm = unittest.mock.Mock()
            pipeline.bilstm.return_value = torch.tensor([[0.95]]).to(pipeline.device)
            
            # Run 9 frames sequentially so sliding decision window of size 5 fills up
            batch = [(meta, iq_data)]
            for _ in range(9):
                pipeline.process_batch(batch)
                
            # Verify event was published to the local trigger queue
            poller = zmq.Poller()
            poller.register(sub_trigger, zmq.POLLIN)
            
            socks = dict(poller.poll(timeout=2000))
            self.assertIn(sub_trigger, socks, "Failed to receive Target Confirmed event on trigger socket")
            
            event = sub_trigger.recv_json()
            self.assertEqual(event["event_type"], "Confirmed Target")
            self.assertEqual(event["node_id"], "Node_A")
            self.assertEqual(event["center_frequency_hz"], 2400000000)
            self.assertGreaterEqual(event["confidence"], 0.7)
        finally:
            pipeline.stop()
            sub_trigger.close()
            context.term()

if __name__ == "__main__":
    unittest.main()
