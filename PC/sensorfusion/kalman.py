import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

import numpy as np
from filterpy.kalman import KalmanFilter

class Kalman:
    dt = 0.06  # seconds

    def __init__(self):
        self.kf = KalmanFilter(dim_x=12, dim_z=4)
        self.kf.x = np.zeros(12)

        dt = self.dt
        dt2 = 0.5 * dt ** 2
        self.kf.F = np.eye(12)
        for i in range(4):
            self.kf.F[i, i + 4] = dt
            self.kf.F[i, i + 8] = dt2
            self.kf.F[i + 4, i + 8] = dt

        self.kf.H = np.zeros((4, 12))
        self.kf.H[0, 0] = 1
        self.kf.H[1, 1] = 1
        self.kf.H[2, 2] = 1
        self.kf.H[3, 3] = 1

        self.kf.P *= 50.

        self.R_bf = np.diag([0.5, 0.8, 0.5, 0.8])  # bounding box noise
        self.R_obj = np.diag([0.05, 0.05, 0.05, 0.05])
        self.R_iou = np.diag([0.01, 0.01, 0.01, 0.01])

        self.kf.Q = self.build_process_noise(self.dt, var=2)

        freq = 1.0 / self.dt
        self.filters = [
            OneEuroFilter(freq, min_cutoff=1.5, beta=0.2),  # more smoothing
            OneEuroFilter(freq, min_cutoff=1.5, beta=0.2),  # vertical - more aggressive
            OneEuroFilter(freq, min_cutoff=1.5, beta=0.2),
            OneEuroFilter(freq, min_cutoff=1.5, beta=0.2),
        ]

        self.missing_counter = 0
        self.max_missing = 5

    def build_process_noise(self, dt, var):
        """Create a 12x12 process noise matrix for constant acceleration."""
        dt2 = dt**2
        dt3 = dt**3 / 2
        dt4 = dt**4 / 4
        q = np.array([
            [dt4, dt3, dt2 / 2],
            [dt3, dt2, dt],
            [dt2 / 2, dt, 1]
        ]) * var

        Q = np.zeros((12, 12))
        for i in range(4):  # for each coordinate (x1, y1, x2, y2)
            Q[i*3:i*3+3, i*3:i*3+3] = q
        return Q

    def predict(self):
        self.kf.predict()
        self.missing_counter += 1

    def sensor_update(self, z, R=None):
        if self.missing_counter >= self.max_missing:
            print("[Kalman] Reacquiring target, inflating P and trusting measurement more")
            self.kf.P *= 3
            R = np.diag([0.01, 0.01, 0.01, 0.01])
        elif R is None:
            R = self.R_obj

        self.kf.R = R
        z = np.array(z, dtype=np.float32)
        if hasattr(self, "last_z"):
            alpha = 0.7  # higher = smoother, lower = reacts faster
            z = alpha * z + (1 - alpha) * self.last_z
        self.last_z = z.copy()

        # Now update
        self.kf.update(z)
        self.missing_counter = 0

    def get_smoothed_state(self, 
                       min_size=30, max_size=150, 
                       min_aspect=0.5, max_aspect=2.0):
        """
        Returns the smoothed bounding box [x1, y1, x2, y2], clamped to size and aspect ratio.

        Args:
            min_size (int): minimum width/height in pixels
            max_size (int): maximum width/height in pixels
            min_aspect (float): minimum width/height ratio (e.g., 0.5 means width >= 0.5*height)
            max_aspect (float): maximum width/height ratio (e.g., 2.0 means width <= 2*height)
        """
        x_raw = self.kf.x[:4]
        smoothed = np.array([f.filter(val, self.dt) for f, val in zip(self.filters, x_raw)])

        x1, y1, x2, y2 = smoothed
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        # Clamp width
        if width < min_size:
            center_x = (x1 + x2) / 2
            width = min_size
            x1 = center_x - width / 2
            x2 = center_x + width / 2
        elif width > max_size:
            center_x = (x1 + x2) / 2
            width = max_size
            x1 = center_x - width / 2
            x2 = center_x + width / 2

        # Clamp height
        if height < min_size:
            center_y = (y1 + y2) / 2
            height = min_size
            y1 = center_y - height / 2
            y2 = center_y + height / 2
        elif height > max_size:
            center_y = (y1 + y2) / 2
            height = max_size
            y1 = center_y - height / 2
            y2 = center_y + height / 2

        # Enforce aspect ratio
        aspect = width / height if height != 0 else 1.0

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        if aspect < min_aspect:
            # Too tall/narrow: increase width
            width = height * min_aspect
            x1 = center_x - width / 2
            x2 = center_x + width / 2
        elif aspect > max_aspect:
            # Too wide/short: increase height
            height = width / max_aspect
            y1 = center_y - height / 2
            y2 = center_y + height / 2

        return np.array([x1, y1, x2, y2])



    def increase_uncertainty(self, measurement=True, process=False, factor=1.1):
        """Increase measurement and/or process noise by a factor"""
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
