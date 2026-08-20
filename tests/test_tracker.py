import os
import sys
import unittest
import numpy as np
import yaml
import time
import zmq

# Add root folder to python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ekf import EKF
from tracker import triangulate_3d_point, MultiTargetTracker, DroneTrack

class TestTrackerAndEKF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = "config.yaml"
        with open(cls.config_path, "r") as f:
            cls.config = yaml.safe_load(f)
            
        cls.node_cfg = cls.config["node"]
        cls.track_cfg = cls.config["tracking"]
        cls.node_pos_A = cls.track_cfg["node_positions"]["Node_A"]
        cls.node_pos_B = cls.track_cfg["node_positions"]["Node_B"]

    def test_line_triangulation(self):
        """
        Verify that least-squares triangulation resolves the intersection midpoint of crossing lines.
        """
        # Node A at (0, 0, 10), Node B at (500, 0, 12)
        # Target at (250, 100, 50)
        target = np.array([250.0, 100.0, 50.0])
        
        # Vectors from nodes to target
        v_A = target - np.array(self.node_pos_A)
        v_B = target - np.array(self.node_pos_B)
        
        # Convert to bearing angles (degrees)
        az_A = np.rad2deg(np.arctan2(v_A[1], v_A[0]))
        el_A = np.rad2deg(np.arctan2(v_A[2], np.linalg.norm(v_A[0:2])))
        
        az_B = np.rad2deg(np.arctan2(v_B[1], v_B[0]))
        el_B = np.rad2deg(np.arctan2(v_B[2], np.linalg.norm(v_B[0:2])))
        
        midpoint = triangulate_3d_point(
            self.node_pos_A, az_A, el_A,
            self.node_pos_B, az_B, el_B
        )
        
        # Assert returned midpoint is within 1 meter of target
        self.assertLess(np.linalg.norm(midpoint - target), 1.0)

    def test_ekf_single_node_update(self):
        """
        Verify that the EKF correctly updates target ranges and positions from a single node's bearing vectors.
        """
        # Target at (100, 100, 20) relative to Node A
        target = np.array([100.0, 100.0, 20.0])
        v_A = target - np.array(self.node_pos_A)
        
        az_A = np.rad2deg(np.arctan2(v_A[1], v_A[0]))
        el_A = np.rad2deg(np.arctan2(v_A[2], np.linalg.norm(v_A[0:2])))
        
        # Initialize EKF at (90, 90, 15) with zero velocity
        init_state = [90.0, 90.0, 15.0, 0.0, 0.0, 0.0]
        init_cov = np.eye(6) * 10.0
        
        ekf = EKF(
            init_state=init_state,
            init_cov=init_cov,
            process_noise_std=self.track_cfg["process_noise"],
            measurement_noise_deg=self.track_cfg["measurement_noise_deg"]
        )
        
        # Update using Node A angles
        ekf.update([az_A, el_A], self.node_pos_A)
        
        # The EKF estimate should pull closer to the true target position (100, 100, 20)
        # Verify Euclidean distance decreased
        initial_dist = np.linalg.norm(np.array(init_state[0:3]) - target)
        updated_dist = np.linalg.norm(ekf.x[0:3] - target)
        
        self.assertLess(updated_dist, initial_dist, "EKF single-node update failed to pull state closer to target!")

    def test_hungarian_association_and_gating(self):
        """
        Verify Hungarian data association and gating threshold checks.
        """
        tracker = MultiTargetTracker(self.config_path)
        
        # Create two mock active tracks
        track1 = DroneTrack(1, [100.0, 100.0, 20.0], 0.1, 2.0)
        track1.state = "active"
        
        track2 = DroneTrack(2, [200.0, 200.0, 30.0], 0.1, 2.0)
        track2.state = "active"
        
        tracker.tracks = [track1, track2]
        
        # Create incoming peaks (measurements from Node A)
        # Peak 1: close to Track 1
        # Peak 2: close to Track 2
        # Peak 3: way out (outlier, should trigger candidate birth but be gated)
        
        # Convert true position of track 1 to angles from Node A
        pos_A = self.node_pos_A
        v1 = np.array([101.0, 100.0, 20.0]) - pos_A # delta x shifted by 1m
        v2 = np.array([199.0, 200.0, 30.0]) - pos_A # delta x shifted by -1m
        v3 = np.array([500.0, 500.0, 100.0]) - pos_A # outlier
        
        peak1 = {
            "azimuth": np.rad2deg(np.arctan2(v1[1], v1[0])),
            "elevation": np.rad2deg(np.arctan2(v1[2], np.linalg.norm(v1[0:2]))),
            "snr": 15.0
        }
        peak2 = {
            "azimuth": np.rad2deg(np.arctan2(v2[1], v2[0])),
            "elevation": np.rad2deg(np.arctan2(v2[2], np.linalg.norm(v2[0:2]))),
            "snr": 15.0
        }
        peak3 = {
            "azimuth": np.rad2deg(np.arctan2(v3[1], v3[0])),
            "elevation": np.rad2deg(np.arctan2(v3[2], np.linalg.norm(v3[0:2]))),
            "snr": 15.0
        }
        
        event = {
            "node_id": "Node_A",
            "timestamp": time.time(),
            "peaks": [peak1, peak2, peak3]
        }
        
        tracker.associate_and_update(event)
        
        # Check that Track 1 was updated and death counter is 0
        self.assertEqual(track1.death_counter, 0)
        self.assertEqual(track2.death_counter, 0)
        
        # Unassociated Peak 3 should have been saved in unassociated list
        self.assertEqual(len(tracker.unassociated_peaks), 1)
        tracker.context.term()

    def test_kinematic_bounds_gating(self):
        """
        Verify that tracks exceeding configured max_velocity_mps are published with degraded confidence.
        """
        tracker = MultiTargetTracker(self.config_path)
        
        # Create track and manually inject high velocity into state
        track = DroneTrack(1, [100.0, 100.0, 20.0], 0.1, 2.0)
        track.state = "active"
        # Set velocity to 100 m/s (exceeds max_velocity_mps = 45.0)
        track.ekf.x[3] = 100.0
        
        tracker.tracks = [track]
        
        # Set up a listener for track updates
        context = zmq.Context()
        sub_tracks = context.socket(zmq.SUB)
        sub_tracks.setsockopt(zmq.SUBSCRIBE, b"")
        sub_tracks.connect(f"tcp://127.0.0.1:{self.track_cfg['tracks_pub_port']}")
        
        # Give ZMQ subscriber time to establish the connection
        time.sleep(0.5)
        
        # Publish
        tracker.publish_tracks(time.time())
        
        # Verify event was published and confidence is degraded to 0.3
        poller = zmq.Poller()
        poller.register(sub_tracks, zmq.POLLIN)
        socks = dict(poller.poll(timeout=1000))
        
        self.assertIn(sub_tracks, socks)
        event = sub_tracks.recv_json()
        self.assertEqual(event["track_id"], 1)
        self.assertEqual(event["confidence"], 0.3)  # Degraded
        
        sub_tracks.close()
        context.term()

    def test_multi_node_ekf_fusion_convergence(self):
        """
        Verify EKF converges on correct 3D trajectory when fed multi-node synthetic observations.
        """
        # Drone moving at constant velocity: Start at (100, 100, 20), moving vx=10, vy=10, vz=1 m/s
        pos = np.array([100.0, 100.0, 20.0])
        vel = np.array([10.0, 10.0, 1.0])
        
        # Initialize EKF at (90, 90, 15) with zero velocity
        init_state = [90.0, 90.0, 15.0, 0.0, 0.0, 0.0]
        init_cov = np.eye(6) * 100.0
        
        ekf = EKF(init_state, init_cov, process_noise_std=0.1, measurement_noise_deg=2.0)
        
        dt = 0.5
        for step in range(10):
            # Move target
            pos = pos + vel * dt
            
            # Predict EKF
            ekf.predict(dt)
            
            # Observations from Node A and Node B
            v_A = pos - np.array(self.node_pos_A)
            az_A = np.rad2deg(np.arctan2(v_A[1], v_A[0]))
            el_A = np.rad2deg(np.arctan2(v_A[2], np.linalg.norm(v_A[0:2])))
            
            v_B = pos - np.array(self.node_pos_B)
            az_B = np.rad2deg(np.arctan2(v_B[1], v_B[0]))
            el_B = np.rad2deg(np.arctan2(v_B[2], np.linalg.norm(v_B[0:2])))
            
            # Sequential EKF Update
            ekf.update([az_A, el_A], self.node_pos_A)
            ekf.update([az_B, el_B], self.node_pos_B)
            
        # Verify estimated position and velocity are close to true trajectory
        est_pos = ekf.x[0:3]
        est_vel = ekf.x[3:6]
        
        print(f"True Final Pos: {pos}, Est Final Pos: {est_pos}")
        print(f"True Final Vel: {vel}, Est Final Vel: {est_vel}")
        
        self.assertLess(np.linalg.norm(est_pos - pos), 5.0, "Multi-node EKF failed to converge on target position!")
        self.assertLess(np.linalg.norm(est_vel - vel), 3.0, "Multi-node EKF failed to converge on target velocity!")

if __name__ == "__main__":
    unittest.main()
