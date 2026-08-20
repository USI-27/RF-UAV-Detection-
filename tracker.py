import os
import sys
import time
import yaml
import json
import zmq
import threading
import numpy as np
from scipy.optimize import linear_sum_assignment
from ekf import EKF

def triangulate_3d_point(pos_A, az_A, el_A, pos_B, az_B, el_B):
    """
    Computes the 3D midpoint of the closest approach between two bearing vectors (least-squares).
    """
    theta_A = np.deg2rad(az_A)
    phi_A = np.deg2rad(el_A)
    theta_B = np.deg2rad(az_B)
    phi_B = np.deg2rad(el_B)
    
    # Unit direction vectors
    v_A = np.array([
        np.cos(phi_A) * np.cos(theta_A),
        np.cos(phi_A) * np.sin(theta_A),
        np.sin(phi_A)
    ])
    v_B = np.array([
        np.cos(phi_B) * np.cos(theta_B),
        np.cos(phi_B) * np.sin(theta_B),
        np.sin(phi_B)
    ])
    
    p_A = np.array(pos_A)
    p_B = np.array(pos_B)
    d = p_A - p_B
    
    a = np.dot(v_A, v_A)
    b = np.dot(v_A, v_B)
    c = np.dot(v_B, v_B)
    
    d_A = np.dot(v_A, d)
    d_B = np.dot(v_B, d)
    
    M = np.array([
        [a, -b],
        [b, -c]
    ])
    rhs = np.array([-d_A, -d_B])
    
    try:
        t, s = np.linalg.solve(M, rhs)
        point_A = p_A + t * v_A
        point_B = p_B + s * v_B
        midpoint = 0.5 * (point_A + point_B)
        return midpoint
    except np.linalg.LinAlgError:
        # Parallel lines fallback
        return 0.5 * (p_A + p_B)

class DroneTrack:
    def __init__(self, track_id, init_pos, process_noise, meas_noise_deg):
        self.track_id = track_id
        
        # Position initialization var = 50m^2, Velocity var = 10m^2/s^2
        P_init = np.diag([50.0, 50.0, 50.0, 10.0, 10.0, 10.0])
        init_state = [init_pos[0], init_pos[1], init_pos[2], 0.0, 0.0, 0.0]
        
        self.ekf = EKF(init_state, P_init, process_noise, meas_noise_deg)
        self.state = "candidate"  # "candidate" | "active" | "retired"
        self.birth_counter = 1
        self.death_counter = 0
        self.last_update_time = time.time()
        self.last_pub_time = 0.0

class MultiTargetTracker:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.node_cfg = self.config["node"]
        self.track_cfg = self.config["tracking"]
        
        self.tracks = []
        self.next_track_id = 1
        
        # Buffer of unassociated measurements from Node A and Node B to perform cross-node triangulation birth
        self.unassociated_peaks = []
        
        # ZMQ setup
        self.context = zmq.Context()
        self.sub_angles = self.context.socket(zmq.SUB)
        self.sub_angles.setsockopt(zmq.SUBSCRIBE, b"")
        self.sub_angles.connect(f"tcp://127.0.0.1:{self.config['geometry_engine']['angles_pub_port']}")
        
        self.pub_tracks = self.context.socket(zmq.PUB)
        self.pub_tracks.setsockopt(zmq.SNDHWM, 1000)
        self.pub_tracks.bind(f"tcp://{self.node_cfg['bind_address']}:{self.track_cfg['tracks_pub_port']}")
        
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        
        self.audit_log_path = "tracker_assignment_audit.jsonl"
        self.max_concurrent_tracks = 20  # Resource protection: cap max phantoms

    def write_audit_log(self, entry):
        entry["system_time"] = time.time()
        try:
            with open(self.audit_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"Error writing tracker audit log: {e}", file=sys.stderr)

    def associate_and_update(self, event):
        """
        Runs optimal assignment using SciPy's Hungarian algorithm.
        """
        node_id = event["node_id"]
        timestamp = event["timestamp"]
        peaks = event["peaks"]
        
        node_pos = self.track_cfg["node_positions"][node_id]
        
        with self.lock:
            active_or_candidate_tracks = [t for t in self.tracks if t.state in ("candidate", "active")]
            
            # Predict step for all tracks based on elapsed time delta
            for track in active_or_candidate_tracks:
                dt = timestamp - track.last_update_time
                if dt > 0:
                    track.ekf.predict(dt)
                    track.last_update_time = timestamp

            # Construct cost matrix: Rows = tracks, Cols = new measurements (peaks)
            num_tracks = len(active_or_candidate_tracks)
            num_meas = len(peaks)
            
            matched_meas_indices = set()
            
            if num_tracks > 0 and num_meas > 0:
                cost_matrix = np.zeros((num_tracks, num_meas))
                for i, track in enumerate(active_or_candidate_tracks):
                    for j, peak in enumerate(peaks):
                        meas_z = [peak["azimuth"], peak["elevation"]]
                        # Cost is Mahalanobis distance squared
                        cost_matrix[i, j] = track.ekf.compute_mahalanobis_distance(meas_z, node_pos)

                # Solve assignment
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                # Apply gate threshold check
                for i_track, j_meas in zip(row_ind, col_ind):
                    cost = cost_matrix[i_track, j_meas]
                    track = active_or_candidate_tracks[i_track]
                    peak = peaks[j_meas]
                    meas_z = [peak["azimuth"], peak["elevation"]]
                    
                    self.write_audit_log({
                        "event": "assignment_decision",
                        "track_id": track.track_id,
                        "node_id": node_id,
                        "cost_mahalanobis": cost,
                        "gate_threshold": self.track_cfg["max_association_distance"],
                        "accepted": cost < self.track_cfg["max_association_distance"]
                    })
                    
                    if cost < self.track_cfg["max_association_distance"]:
                        # 1. Innovation Gating (Mahalanobis Check)
                        # chi2 gating matches are already filtered by the cost limit above
                        track.ekf.update(meas_z, node_pos)
                        track.death_counter = 0
                        matched_meas_indices.add(j_meas)
                        
                        # Promote candidate track to active
                        if track.state == "candidate":
                            track.birth_counter += 1
                            if track.birth_counter >= self.track_cfg["birth_count_threshold"]:
                                track.state = "active"
                    else:
                        # Cost too high: track counts as missed frame
                        track.death_counter += 1

                # Update death counter for unmatched tracks in this event
                matched_track_indices = set(row_ind)
                for idx, track in enumerate(active_or_candidate_tracks):
                    if idx not in matched_track_indices:
                        track.death_counter += 1

            else:
                # No tracks existed: all measurements are unmatched
                for track in active_or_candidate_tracks:
                    track.death_counter += 1

            # Retain unmatched peaks for track birth / cross-node triangulation
            for j, peak in enumerate(peaks):
                if j not in matched_meas_indices:
                    self.unassociated_peaks.append({
                        "node_id": node_id,
                        "peak": peak,
                        "timestamp": timestamp
                    })

            # Perform track birth (cross-node triangulation checks)
            self.process_track_births()

            # Clean up old tracks (death counter threshold)
            for track in self.tracks:
                if track.state in ("candidate", "active"):
                    # Coast on prediction for up to death_count_threshold
                    if track.death_counter >= self.track_cfg["death_count_threshold"]:
                        track.state = "retired"
                        self.write_audit_log({
                            "event": "track_retired",
                            "track_id": track.track_id
                        })

            # Publish updated active tracks
            self.publish_tracks(timestamp)

    def process_track_births(self):
        """
        Security check: require multi-node corroboration (Node A & Node B within 100ms)
        to spawn a new candidate track. This mitigates track flooding DoS attacks.
        """
        now = time.time()
        # Keep unassociated peaks only within last 2 seconds
        self.unassociated_peaks = [p for p in self.unassociated_peaks if now - p["timestamp"] < 2.0]
        
        # Separate peaks by node
        peaks_A = [p for p in self.unassociated_peaks if p["node_id"] == "Node_A"]
        peaks_B = [p for p in self.unassociated_peaks if p["node_id"] == "Node_B"]
        
        used_A = set()
        used_B = set()
        
        for pa in peaks_A:
            for pb in peaks_B:
                # Check timing similarity (within 100ms)
                if abs(pa["timestamp"] - pb["timestamp"]) < 0.1:
                    pos_A = self.track_cfg["node_positions"]["Node_A"]
                    pos_B = self.track_cfg["node_positions"]["Node_B"]
                    
                    # Triangulate 3D point
                    init_pos = triangulate_3d_point(
                        pos_A, pa["peak"]["azimuth"], pa["peak"]["elevation"],
                        pos_B, pb["peak"]["azimuth"], pb["peak"]["elevation"]
                    )
                    
                    # Security Check: Cap maximum concurrent tracks to prevent memory flooding DoS
                    active_count = sum(1 for t in self.tracks if t.state in ("candidate", "active"))
                    if active_count >= self.max_concurrent_tracks:
                        print("WARNING: Track cap reached. Dropping track candidate birth.", file=sys.stderr)
                        return
                    
                    # Check if this initial point is close to an existing track to avoid spawning duplicates
                    duplicate = False
                    for track in self.tracks:
                        if track.state in ("candidate", "active"):
                            dist = np.linalg.norm(track.ekf.x[0:3] - init_pos)
                            if dist < 50.0:  # within 50 meters
                                duplicate = True
                                break
                                
                    if not duplicate:
                        new_track = DroneTrack(
                            track_id=self.next_track_id,
                            init_pos=init_pos,
                            process_noise=self.track_cfg["process_noise"],
                            meas_noise_deg=self.track_cfg["measurement_noise_deg"]
                        )
                        self.tracks.append(new_track)
                        self.write_audit_log({
                            "event": "track_birth",
                            "track_id": self.next_track_id,
                            "position": list(init_pos)
                        })
                        self.next_track_id += 1
                        
                    used_A.add(id(pa))
                    used_B.add(id(pb))
                    break
                    
        # Remove used peaks from buffer
        self.unassociated_peaks = [p for p in self.unassociated_peaks if id(p) not in used_A and id(p) not in used_B]

    def publish_tracks(self, timestamp):
        """
        Publishes all active EKF tracks to the downstream consumer layer (UI).
        """
        for track in self.tracks:
            if track.state == "active":
                x = track.ekf.x
                vel_magnitude = float(np.linalg.norm(x[3:6]))
                
                # Check physical kinematic limits (max velocity bound check)
                is_implausible = vel_magnitude > self.track_cfg["max_velocity_mps"]
                confidence = 0.3 if is_implausible else 0.95
                
                if is_implausible:
                    self.write_audit_log({
                        "event": "kinematic_bound_violated",
                        "track_id": track.track_id,
                        "velocity_mps": vel_magnitude
                    })
                
                # Determine mode
                mode = "coasting" if track.death_counter > 0 else "active"
                
                track_event = {
                    "track_id": track.track_id,
                    "x": float(x[0]),
                    "y": float(x[1]),
                    "z": float(x[2]),
                    "vx": float(x[3]),
                    "vy": float(x[4]),
                    "vz": float(x[5]),
                    "velocity_mps": vel_magnitude,
                    "mode": mode,
                    "confidence": confidence,
                    "timestamp": timestamp
                }
                
                # Emit
                self.pub_tracks.send_json(track_event)

    def run(self):
        print("Multi-Target Tracker service started...")
        poller = zmq.Poller()
        poller.register(self.sub_angles, zmq.POLLIN)
        
        while not self.stop_event.is_set():
            socks = dict(poller.poll(timeout=100))
            if self.sub_angles in socks:
                try:
                    event = self.sub_angles.recv_json()
                    self.associate_and_update(event)
                except Exception as e:
                    print(f"Error in tracker subscription loop: {e}", file=sys.stderr)
                    
        self.sub_angles.close()
        self.pub_tracks.close()
        self.context.term()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tracker.py <config_path>")
        sys.exit(1)
        
    cfg_path = sys.argv[1]
    tracker = MultiTargetTracker(cfg_path)
    
    try:
        tracker.run()
    except KeyboardInterrupt:
        print("Shutting down Multi-Target Tracker...")
        tracker.stop_event.set()
