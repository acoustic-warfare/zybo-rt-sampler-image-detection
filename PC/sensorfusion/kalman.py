import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class Kalman:
    dt = 0.06  # seconds

    def __init__(self):

        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.x = np.zeros(8)
        self.kf.F = np.eye(8)
        for i in range(4):
            self.kf.F[i, i+4] = self.dt

        self.kf.H = np.zeros((4, 8))
        self.kf.H[0, 0] = 1
        self.kf.H[1, 1] = 1
        self.kf.H[2, 2] = 1
        self.kf.H[3, 3] = 1

        self.kf.P *= 50.
        self.R_bf = np.diag([0.2, 0.2, 0.2, 0.2])  # Bounding box measurement noise
        self.R_obj = np.diag([0.05, 0.05, 0.05, 0.05])
        self.R_iou = np.diag([0.01, 0.01, 0.01, 0.01])

        q = Q_discrete_white_noise(dim=2, dt=self.dt, var=10)
        self.kf.Q = np.zeros((8, 8))
        self.kf.Q[:2, :2] = q
        self.kf.Q[2:4, 2:4] = q
        self.kf.Q[4:6, 4:6] = q
        self.kf.Q[6:8, 6:8] = q

        # 1 Euro filters for each bounding box coordinate
        freq = 1.0 / self.dt
        self.filters = [
            OneEuroFilter(freq, min_cutoff=1.0, beta=0.3),
            OneEuroFilter(freq, min_cutoff=1.0, beta=0.3),
            OneEuroFilter(freq, min_cutoff=1.0, beta=0.3),
            OneEuroFilter(freq, min_cutoff=1.0, beta=0.3),
        ]
        self.missing_counter = 0
        self.max_missing = 5  # Or however many frames you expect to tolerate

    def predict(self):
        self.kf.predict()
        self.missing_counter += 1

    def sensor_update(self, z, R=None):
        # Optionally inflate P if we've been missing for a while
        if self.missing_counter >= self.max_missing:
            print("[Kalman] Reacquiring target, inflating P and trusting measurement more")
            self.kf.P *= 3  # Inflate uncertainty so it can snap
            R = np.diag([0.01, 0.01, 0.01, 0.01])  # Trust this measurement a lot
        elif R is None:
            R = self.R_obj  # Default

        self.kf.R = R
        self.kf.update(z)
        self.missing_counter = 0  # Reset on successful measurement

    def get_smoothed_state(self):
        """Returns the filtered bounding box: x1, y1, x2, y2 (smoothed by 1 Euro)"""
        x_raw = self.kf.x[:4]
        return np.array([f.filter(val, self.dt) for f, val in zip(self.filters, x_raw)])
    
    def increase_uncertainty(self, measurement=True, process=False, factor=1.1):
        """Increase measurement and/or process noise by a factor (default 10%).

        Args:
            measurement (bool): Whether to increase measurement noise (R)
            process (bool): Whether to increase process noise (Q)
            factor (float): Multiplier for increasing noise (e.g., 1.1 = +10%)
        """
        if measurement:
            self.kf.R *= factor
        if process:
            self.kf.Q *= factor




import math

class OneEuroFilter:
    def __init__(self, freq, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0

    def alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x, dt):
        if self.x_prev is None:
            self.x_prev = x
            return x

        dx = (x - self.x_prev) / dt
        alpha_d = self.alpha(self.d_cutoff)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self.alpha(cutoff)

        x_hat = alpha * x + (1 - alpha) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat

        return x_hat
