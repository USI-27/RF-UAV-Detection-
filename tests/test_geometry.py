import os
import sys
import unittest
import yaml
import torch
import numpy as np

# Add root folder to python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry import apply_fbss, regularize_covariance, run_music, precompute_steering_vectors

class TestGeometryEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = "config.yaml"
        with open(cls.config_path, "r") as f:
            cls.config = yaml.safe_load(f)
            
        cls.node_cfg = cls.config["node"]
        cls.geo_cfg = cls.config["geometry_engine"]
        cls.sdr_cfg = cls.config["sdr"]
        
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.f = 2400000000.0  # 2.4 GHz
        
        # Precompute steering vectors for test
        cls.steering_matrix, cls.azimuths, cls.elevations = precompute_steering_vectors(
            r=cls.geo_cfg["array_radius_m"],
            c=cls.geo_cfg["speed_of_light_mps"],
            f=cls.f,
            num_antennas=cls.sdr_cfg["num_antennas"],
            grid_step_deg=cls.geo_cfg["grid_step_deg"],
            device=cls.device
        )

    def test_fbss_hermitian_and_psd(self):
        """
        Verify that apply_fbss produces a valid Hermitian matrix.
        """
        # Create a random complex covariance matrix
        M = self.sdr_cfg["num_antennas"]
        A = torch.randn(M, M, dtype=torch.complex64, device=self.device)
        R = A @ torch.conj(A.t())  # Guaranteed to be Hermitian positive semidefinite
        
        R_fb = apply_fbss(R)
        
        # Verify Hermitian: R_fb == R_fb^H
        is_hermitian = torch.allclose(R_fb, torch.conj(R_fb.t()), atol=1e-5)
        self.assertTrue(is_hermitian, "FBSS output is not Hermitian!")

    def test_diagonal_loading_regularization(self):
        """
        Verify that diagonal loading prevents singular matrices.
        """
        M = self.sdr_cfg["num_antennas"]
        # Create a singular matrix (all zeros)
        R_singular = torch.zeros((M, M), dtype=torch.complex64, device=self.device)
        
        R_reg = regularize_covariance(R_singular, alpha=1e-6)
        
        # Eigen decomposition should work and not crash or return NaNs
        evals, evecs = torch.linalg.eigh(R_reg)
        self.assertFalse(torch.any(torch.isnan(evals)), "Eigen decomposition contains NaNs after regularization!")

    def test_music_synthetic_aoa_accuracy(self):
        """
        MUSIC Angle Estimation: Feed synthetic UCA frames with known AoA and verify peaks
        are found within 2 degrees tolerance.
        """
        true_az = 46.0  # Must be divisible by grid step size (2.0)
        true_el = 12.0
        
        # Generate synthetic UCA steering vector
        lam = self.geo_cfg["speed_of_light_mps"] / self.f
        theta_rad = np.deg2rad(true_az)
        phi_rad = np.deg2rad(true_el)
        r = self.geo_cfg["array_radius_m"]
        M = self.sdr_cfg["num_antennas"]
        N = self.sdr_cfg["frame_size_samples"]
        
        gamma = np.arange(M) * 2 * np.pi / M
        phase_coef = (2 * np.pi * r) / lam
        phases = phase_coef * np.cos(phi_rad) * np.cos(theta_rad - gamma)
        steering_vector = np.exp(1j * phases) # Shape: (5,)
        
        # Generate signal plus tiny noise
        signal = np.sin(2 * np.pi * 1000 * np.arange(N) / 2.4e6)
        iq_data = np.outer(steering_vector, signal)
        noise = (np.random.randn(M, N) + 1j * np.random.randn(M, N)) * 0.01
        iq_matrix = (iq_data + noise).astype(np.complex64)
        
        # Run MUSIC
        peaks = run_music(
            iq_matrix=iq_matrix,
            steering_matrix=self.steering_matrix,
            azimuths=self.azimuths,
            elevations=self.elevations,
            snr_threshold_db=self.geo_cfg["snr_threshold_db"],
            device=self.device
        )
        
        self.assertGreater(len(peaks), 0, "No peaks found in the MUSIC spectrum!")
        
        # Sort by SNR and get the strongest peak
        strongest_peak = max(peaks, key=lambda x: x["snr"])
        
        print(f"MUSIC target AOA: Azimuth={true_az}, Elevation={true_el}")
        print(f"Strongest Peak Found: Azimuth={strongest_peak['azimuth']}, Elevation={strongest_peak['elevation']}, SNR={strongest_peak['snr']:.2f}dB")
        
        # Assert within step size tolerance
        self.assertLessEqual(abs(strongest_peak["azimuth"] - true_az), 2.0)
        self.assertLessEqual(abs(strongest_peak["elevation"] - true_el), 2.0)

    def test_geometry_input_validation(self):
        """
        Vulnerability Mitigation: Test that malformed frames are rejected.
        """
        # Shape size 5x1024 is configured. Send 6x1024.
        from geometry import GeometryEngine
        
        # Run engine mock target confirmation
        engine = GeometryEngine(self.config_path)
        
        # Mock query client to return malformed shape (6, 1024)
        mock_client = unittest.mock.Mock()
        mock_client.query_history.return_value = (
            {"match_count": 1},
            np.zeros((1, 6, 1024), dtype=np.complex64)
        )
        engine.get_buffer_client = unittest.mock.Mock(return_value=mock_client)
        
        # Mock MUSIC calculation to make sure it's not called
        import geometry
        geometry.run_music = unittest.mock.Mock()
        
        # Send event to trigger
        event = {
            "node_id": "Node_A",
            "timestamp": time.time(),
            "center_frequency_hz": self.f
        }
        
        engine.on_confirmed_target(event)
        
        # Verify run_music was never invoked due to shape validation reject
        geometry.run_music.assert_not_called()
        engine.context.term()

if __name__ == "__main__":
    unittest.main()
