import numpy as np

class EKF:
    """
    Extended Kalman Filter for 3D trajectory tracking.
    State vector x: [x, y, z, vx, vy, vz]^T
    """
    def __init__(self, init_state, init_cov, process_noise_std, measurement_noise_deg):
        self.x = np.array(init_state, dtype=np.float64)  # Shape (6,)
        self.P = np.array(init_cov, dtype=np.float64)    # Shape (6, 6)
        
        self.q = process_noise_std
        # Convert measurement noise to radians
        self.r_val = np.deg2rad(measurement_noise_deg)
        self.R = np.eye(2) * (self.r_val ** 2)

    def predict(self, dt):
        """
        Predict state and covariance forward in time by dt seconds.
        """
        # State transition matrix F
        F = np.eye(6)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        
        # Propagate state
        self.x = F @ self.x
        
        # Process noise covariance Q (continuous white noise model)
        Q = np.zeros((6, 6))
        Q[0:3, 0:3] = (dt**3 / 3.0) * np.eye(3) * self.q
        Q[0:3, 3:6] = (dt**2 / 2.0) * np.eye(3) * self.q
        Q[3:6, 0:3] = (dt**2 / 2.0) * np.eye(3) * self.q
        Q[3:6, 3:6] = dt * np.eye(3) * self.q
        
        # Propagate covariance
        self.P = F @ self.P @ F.T + Q
        
        # Numerical safeguard: bound covariance growth / guarantee symmetry
        self.P = 0.5 * (self.P + self.P.T)
        # Prevent numerical underflow in diagonal to avoid singular matrices
        min_var = 1e-6
        for i in range(6):
            if self.P[i, i] < min_var:
                self.P[i, i] = min_var

    def _get_h_and_jacobian(self, node_pos):
        """
        Computes measurement vector h(x) and Jacobian H for a specific node position.
        """
        dx = self.x[0] - node_pos[0]
        dy = self.x[1] - node_pos[1]
        dz = self.x[2] - node_pos[2]
        
        rho = np.sqrt(dx**2 + dy**2)
        d_range = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Safeguard division by zero
        if rho < 1e-4:
            rho = 1e-4
        if d_range < 1e-4:
            d_range = 1e-4
            
        # Predicted measurements
        pred_az = np.arctan2(dy, dx)
        pred_el = np.arctan2(dz, rho)
        h_x = np.array([pred_az, pred_el])
        
        # Compute Jacobian H (shape 2x6)
        H = np.zeros((2, 6))
        
        # Azimuth derivatives
        H[0, 0] = -dy / (rho**2)
        H[0, 1] = dx / (rho**2)
        H[0, 2] = 0.0
        
        # Elevation derivatives
        H[1, 0] = -dx * dz / ((d_range**2) * rho)
        H[1, 1] = -dy * dz / ((d_range**2) * rho)
        H[1, 2] = rho / (d_range**2)
        
        # Velocity derivatives are zero
        H[0, 3:6] = 0.0
        H[1, 3:6] = 0.0
        
        return h_x, H

    def get_innovation_and_covariance(self, meas_z, node_pos):
        """
        Returns innovation vector y, Jacobian H, and innovation covariance S.
        """
        # Convert measurement to radians: meas_z = [azimuth_deg, elevation_deg]
        z_rad = np.deg2rad(meas_z)
        
        h_x, H = self._get_h_and_jacobian(node_pos)
        
        # Innovation y
        y = z_rad - h_x
        
        # Circular wrap-around check for Azimuth (innovation[0])
        y[0] = (y[0] + np.pi) % (2 * np.pi) - np.pi
        
        # Innovation covariance S
        S = H @ self.P @ H.T + self.R
        
        return y, H, S

    def compute_mahalanobis_distance(self, meas_z, node_pos):
        """
        Calculates Mahalanobis distance squared of the measurement.
        """
        y, _, S = self.get_innovation_and_covariance(meas_z, node_pos)
        try:
            S_inv = np.linalg.inv(S)
            return y.T @ S_inv @ y
        except np.linalg.LinAlgError:
            return float('inf')

    def update(self, meas_z, node_pos):
        """
        Updates the EKF state with a single node's angular measurement.
        """
        y, H, S = self.get_innovation_and_covariance(meas_z, node_pos)
        
        try:
            S_inv = np.linalg.inv(S)
            K = self.P @ H.T @ S_inv  # Kalman Gain
            
            # Update state
            self.x = self.x + K @ y
            
            # Update covariance (Joseph form for numerical stability)
            I_KH = np.eye(6) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
            
            # Guarantee symmetry
            self.P = 0.5 * (self.P + self.P.T)
        except np.linalg.LinAlgError:
            print("WARNING: Matrix inversion failure in EKF update step.", file=sys.stderr)
