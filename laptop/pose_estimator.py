"""
Pose estimator: Extended Kalman Filter fusing encoder odometry + phone IMU.

State: [x, y, theta]

Prediction step (every tick):
  - Convert 4 encoder delta-ticks to robot-frame (dx, dy, dtheta) via
    mecanum forward kinematics
  - Apply as motion model

Update step (when IMU data available):
  - IMU gyro_z gives measured yaw rate
  - Corrects theta estimate

Optional update step (when scan-match available):
  - LiDAR ICP gives (x, y, theta) correction against occupancy grid
  - Corrects full pose
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np

from config import load_config

logger = logging.getLogger(__name__)


class PoseEstimator:
    """3-state EKF for wheelchair pose: [x, y, theta]."""

    def __init__(self):
        cfg = load_config()
        self._mecanum = cfg.mecanum
        self._encoder_tpr = cfg.serial.encoder_ticks_per_rev
        self._ekf_cfg = cfg.ekf

        # Derived geometry
        self._r = self._mecanum.wheel_radius_m
        self._lx = self._mecanum.wheelbase_m / 2.0   # half wheelbase
        self._ly = self._mecanum.track_width_m / 2.0  # half track width

        # EKF state
        self._x = np.zeros(3)          # [x, y, theta]
        self._P = np.eye(3) * 0.01     # covariance — start near-certain at origin

        # Process noise
        q_xy = self._ekf_cfg.process_noise_xy
        q_th = self._ekf_cfg.process_noise_theta
        self._Q = np.diag([q_xy ** 2, q_xy ** 2, q_th ** 2])

        self._initialized = True

    # ------------------------------------------------------------------
    # Mecanum forward kinematics
    # ------------------------------------------------------------------

    def _ticks_to_velocities(self, ticks: Tuple[int, int, int, int],
                              dt: float) -> Tuple[float, float, float]:
        """Convert 4 encoder delta-ticks to robot-frame (vx, vy, omega).

        Mecanum FK (inverse of the IK in Arduino):
          vx    = (r/4) * (w_FL + w_FR + w_RL + w_RR)
          vy    = (r/4) * (-w_FL + w_FR + w_RL - w_RR)
          omega = (r / (4*(lx+ly))) * (-w_FL + w_FR - w_RL + w_RR)

        where w_i = wheel angular velocity in rad/s = (ticks / tpr) * 2*pi / dt
        """
        if dt <= 0:
            return (0.0, 0.0, 0.0)

        # Convert ticks to wheel angular velocities (rad/s)
        scale = (2.0 * math.pi) / (self._encoder_tpr * dt)
        w = [t * scale for t in ticks]  # [w_FL, w_FR, w_RL, w_RR]

        r = self._r
        vx    = (r / 4.0) * (w[0] + w[1] + w[2] + w[3])
        vy    = (r / 4.0) * (-w[0] + w[1] + w[2] - w[3])
        omega = (r / (4.0 * (self._lx + self._ly))) * (-w[0] + w[1] - w[2] + w[3])

        return (vx, vy, omega)

    # ------------------------------------------------------------------
    # EKF predict
    # ------------------------------------------------------------------

    def predict(self, ticks: Tuple[int, int, int, int], dt: float) -> None:
        """Prediction step using encoder odometry.

        Args:
            ticks: (FL, FR, RL, RR) delta encoder ticks since last call
            dt: time step in seconds
        """
        vx, vy, omega = self._ticks_to_velocities(ticks, dt)

        theta = self._x[2]
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # Robot-frame displacement → world frame
        dx_world = (cos_t * vx - sin_t * vy) * dt
        dy_world = (sin_t * vx + cos_t * vy) * dt
        dtheta = omega * dt

        # State prediction
        self._x[0] += dx_world
        self._x[1] += dy_world
        self._x[2] += dtheta
        self._x[2] = self._normalize_angle(self._x[2])

        # Track omega for IMU cross-check
        self._last_omega_val = omega

        # Jacobian of motion model w.r.t. state
        F = np.eye(3)
        F[0, 2] = (-sin_t * vx - cos_t * vy) * dt
        F[1, 2] = ( cos_t * vx - sin_t * vy) * dt

        # Covariance prediction
        self._P = F @ self._P @ F.T + self._Q

    # ------------------------------------------------------------------
    # EKF update: IMU gyro
    # ------------------------------------------------------------------

    def update_imu(self, gyro_z: float, dt: float) -> None:
        """Update theta using phone IMU yaw rate measurement.

        The gyro provides an independent heading rate. We integrate it and
        compare against the encoder-predicted heading to produce a standard
        EKF correction on theta.

        Args:
            gyro_z: measured yaw rate in rad/s
            dt: time step in seconds
        """
        if dt <= 0:
            return

        # The predict step already applied encoder-based omega*dt to theta.
        # The gyro says the *true* heading change was gyro_z * dt.
        # So the gyro-implied absolute theta is:
        #   theta_gyro = theta_before_predict + gyro_z * dt
        # Which equals:
        #   theta_current - encoder_dtheta + gyro_dtheta
        #   = self._x[2] - _last_omega * dt + gyro_z * dt
        encoder_omega = self._get_last_omega()
        theta_measured = self._x[2] + (gyro_z - encoder_omega) * dt

        # Observation: theta (H = [0, 0, 1])
        H = np.array([[0, 0, 1]])

        r_gyro = self._ekf_cfg.imu_gyro_noise ** 2
        R = np.array([[r_gyro]])

        z = np.array([theta_measured])
        h = np.array([self._x[2]])

        innovation = z - h
        innovation[0] = self._normalize_angle(innovation[0])

        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.solve(S, np.eye(S.shape[0]))

        self._x = self._x + (K @ innovation).flatten()
        self._x[2] = self._normalize_angle(self._x[2])
        self._P = (np.eye(3) - K @ H) @ self._P

    # ------------------------------------------------------------------
    # EKF update: LiDAR scan match (optional, called from slam.py)
    # ------------------------------------------------------------------

    def update_scan_match(self, x_corr: float, y_corr: float, theta_corr: float) -> None:
        """Full pose correction from LiDAR ICP scan matching.

        Args:
            x_corr, y_corr, theta_corr: absolute pose from scan matcher
        """
        H = np.eye(3)
        z = np.array([x_corr, y_corr, theta_corr])
        h = self._x.copy()
        h[2] = self._normalize_angle(h[2])

        r_xy = self._ekf_cfg.lidar_scan_match_noise_xy ** 2
        r_th = self._ekf_cfg.lidar_scan_match_noise_theta ** 2
        R = np.diag([r_xy, r_xy, r_th])

        innovation = z - h
        innovation[2] = self._normalize_angle(innovation[2])

        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.solve(S, np.eye(S.shape[0]))

        self._x = self._x + K @ innovation
        self._x[2] = self._normalize_angle(self._x[2])
        self._P = (np.eye(3) - K @ H) @ self._P

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_pose(self) -> Tuple[float, float, float]:
        """Return (x, y, theta) in world frame."""
        return (float(self._x[0]), float(self._x[1]), float(self._x[2]))

    def get_covariance(self) -> np.ndarray:
        return self._P.copy()

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> None:
        self._x = np.array([x, y, theta])
        self._P = np.eye(3) * 0.01

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    _last_omega_val: float = 0.0

    def _get_last_omega(self) -> float:
        return self._last_omega_val

    @staticmethod
    def _normalize_angle(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))
