"""
Central configuration for the autonomous wheelchair system.
Every tunable parameter lives here. No magic numbers elsewhere.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    _workspace_root = Path(__file__).resolve().parent.parent
    _env_path = _workspace_root / ".env"
    _env_local_path = _workspace_root / ".env.local"
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
    if _env_local_path.exists():
        load_dotenv(_env_local_path, override=True)
except ImportError:
    pass


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float_list(name: str) -> list[float]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [float(item) for item in parsed]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    try:
        return [float(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError:
        return []


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class SerialConfig:
    port: str = "COM3"
    baud: int = 115200
    watchdog_timeout_ms: int = 500
    encoder_ticks_per_rev: int = 1440   # quadrature encoder CPR


@dataclass
class MotorConfig:
    max_pwm: int = 255
    current_sense_threshold: int = 900


@dataclass
class MecanumConfig:
    wheel_radius_m: float = 0.05
    track_width_m: float = 0.45       # left-right center-to-center
    wheelbase_m: float = 0.45         # front-back center-to-center


@dataclass
class VelocityConfig:
    max_vx: float = 0.5
    max_vy: float = 0.2
    max_omega: float = 1.0
    max_accel: float = 0.5
    max_jerk: float = 1.0


@dataclass
class LidarConfig:
    port: str = "COM4"                # separate from Arduino
    baud: int = 230400                # STL-19P / LD19 protocol
    scan_rate_hz: float = 10.0
    min_range_m: float = 0.02
    max_range_m: float = 12.0
    angle_offset_deg: float = 0.0     # rotate scan to match robot frame


@dataclass
class SLAMConfig:
    map_size_m: float = 20.0
    resolution_m: float = 0.05        # 5 cm cells
    pose_node_spacing_m: float = 0.5
    icp_max_iterations: int = 50
    icp_tolerance: float = 0.001
    loop_closure_min_score: float = 0.7


@dataclass
class PlannerConfig:
    replan_interval_s: float = 2.0
    dwa_time_horizon_s: float = 1.0
    dwa_num_samples: int = 100
    potential_field_range_m: float = 1.5
    dwa_weight: float = 0.7
    potential_field_weight: float = 0.3
    goal_tolerance_m: float = 0.15


@dataclass
class SafetyConfig:
    emergency_stop_distance_m: float = 0.30
    watchdog_timeout_ms: int = 500
    current_sense_threshold: int = 900
    max_pwm_limit: int = 200


@dataclass
class CameraConfig:
    left_index: int = 0
    right_index: int = 1
    width: int = 640
    height: int = 480
    fps: int = 15
    orb_features: int = 500
    landmark_match_threshold: float = 50.0   # Hamming distance
    obstacle_cluster_radius_m: float = 0.4


@dataclass
class EKFConfig:
    # Process noise (how much we trust the motion model)
    process_noise_xy: float = 0.01       # m per step
    process_noise_theta: float = 0.005   # rad per step
    # Measurement noise
    imu_gyro_noise: float = 0.02         # rad/s
    lidar_scan_match_noise_xy: float = 0.05
    lidar_scan_match_noise_theta: float = 0.02


@dataclass
class SpatialMemoryConfig:
    map_dir: str = "data/maps"
    landmark_dir: str = "data/landmarks"
    recording_dir: str = "data/recordings"
    dynamic_prior_alpha: float = 0.3
    cost_success_factor: float = 0.9
    cost_failure_factor: float = 1.5
    cost_min: float = 0.1
    cost_max: float = 10.0
    costmap_weights: dict = field(default_factory=lambda: {
        "static": 1.0,
        "dynamic": 0.5,
        "learned": 0.3,
    })


@dataclass
class VoiceConfig:
    api_key: str = field(default_factory=lambda: os.environ.get("ELEVENLABS_API_KEY", ""))
    voice_id: str = "gUABw7pXQjhjt0kNFBTF"
    stt_fallback: str = "whisper"
    tts_fallback: str = "pyttsx3"


@dataclass
class ArmConfig:
    enabled: bool = field(default_factory=lambda: _env_bool("VIAM_ARM_ENABLED", False))
    use_fake: bool = field(default_factory=lambda: _env_bool("VIAM_USE_FAKE_ARM", False))
    address: str = field(default_factory=lambda: os.environ.get("VIAM_MACHINE_ADDRESS", ""))
    api_key: str = field(default_factory=lambda: os.environ.get("VIAM_API_KEY", ""))
    api_key_id: str = field(default_factory=lambda: os.environ.get("VIAM_API_KEY_ID", ""))
    arm_name: str = field(default_factory=lambda: os.environ.get("VIAM_ARM_NAME", "arm"))
    gripper_name: str = field(default_factory=lambda: os.environ.get("VIAM_GRIPPER_NAME", "gripper"))
    default_timeout_s: float = field(default_factory=lambda: _env_float("VIAM_ARM_TIMEOUT_S", 15.0))
    home_joint_positions: list[float] = field(
        default_factory=lambda: _env_float_list("VIAM_ARM_HOME_JOINTS")
    )
    default_frame: str = field(default_factory=lambda: os.environ.get("VIAM_ARM_FRAME", "world"))


@dataclass
class PickupConfig:
    enabled: bool = field(default_factory=lambda: _env_bool("VIAM_PICKUP_ENABLED", False))
    use_fake_vision: bool = field(default_factory=lambda: _env_bool("VIAM_USE_FAKE_VISION", False))
    vision_service_name: str = field(default_factory=lambda: os.environ.get("VIAM_VISION_SERVICE_NAME", ""))
    camera_name: str = field(default_factory=lambda: os.environ.get("VIAM_CAMERA_NAME", ""))
    target_label: str = field(default_factory=lambda: os.environ.get("VIAM_PICKUP_TARGET_LABEL", ""))
    minimum_confidence: float = field(default_factory=lambda: _env_float("VIAM_PICKUP_MIN_CONFIDENCE", 0.55))
    poll_interval_s: float = field(default_factory=lambda: _env_float("VIAM_PICKUP_POLL_INTERVAL_S", 0.75))
    known_ball_diameter_m: float = field(default_factory=lambda: _env_float("VIAM_PICKUP_BALL_DIAMETER_M", 0.067))
    calibration_distance_m: float = field(default_factory=lambda: _env_float("VIAM_PICKUP_CALIBRATION_DISTANCE_M", 0.45))
    calibration_bbox_width_px: float = field(default_factory=lambda: _env_float("VIAM_PICKUP_CALIBRATION_BBOX_WIDTH_PX", 120.0))
    camera_horizontal_fov_deg: float = field(default_factory=lambda: _env_float("VIAM_PICKUP_CAMERA_FOV_DEG", 62.0))
    center_tolerance_px: float = field(default_factory=lambda: _env_float("VIAM_PICKUP_CENTER_TOLERANCE_PX", 40.0))
    pickup_zone_min_range_m: float = field(default_factory=lambda: _env_float("VIAM_PICKUP_MIN_RANGE_M", 0.20))
    pickup_zone_max_range_m: float = field(default_factory=lambda: _env_float("VIAM_PICKUP_MAX_RANGE_M", 0.65))
    pickup_zone_max_lateral_m: float = field(default_factory=lambda: _env_float("VIAM_PICKUP_MAX_LATERAL_M", 0.12))
    arm_forward_offset_m: float = field(default_factory=lambda: _env_float("VIAM_PICKUP_ARM_FORWARD_OFFSET_M", -0.02))
    arm_lateral_offset_m: float = field(default_factory=lambda: _env_float("VIAM_PICKUP_ARM_LATERAL_OFFSET_M", 0.0))
    arm_vertical_offset_m: float = field(default_factory=lambda: _env_float("VIAM_PICKUP_ARM_VERTICAL_OFFSET_M", 0.0))
    approach_standoff_m: float = field(default_factory=lambda: _env_float("VIAM_PICKUP_APPROACH_STANDOFF_M", 0.04))
    pregrasp_height_m: float = field(default_factory=lambda: _env_float("VIAM_PICKUP_PREGRASP_HEIGHT_M", 0.14))
    grasp_height_m: float = field(default_factory=lambda: _env_float("VIAM_PICKUP_GRASP_HEIGHT_M", 0.05))
    lift_height_m: float = field(default_factory=lambda: _env_float("VIAM_PICKUP_LIFT_HEIGHT_M", 0.18))
    max_retries: int = field(default_factory=lambda: _env_int("VIAM_PICKUP_MAX_RETRIES", 1))
    home_after_pickup: bool = field(default_factory=lambda: _env_bool("VIAM_PICKUP_HOME_AFTER", True))


@dataclass
class LLMConfig:
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    model: str = "claude-sonnet-4-6"
    openai_model: str = "gpt-4o-mini"


@dataclass
class FoxgloveConfig:
    host: str = "0.0.0.0"
    port: int = 8765


@dataclass
class PhoneIMUConfig:
    udp_port: int = 5555
    gyro_axis: str = "z"              # which phone axis = yaw


@dataclass
class Config:
    serial: SerialConfig = field(default_factory=SerialConfig)
    motor: MotorConfig = field(default_factory=MotorConfig)
    mecanum: MecanumConfig = field(default_factory=MecanumConfig)
    velocity: VelocityConfig = field(default_factory=VelocityConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    slam: SLAMConfig = field(default_factory=SLAMConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    ekf: EKFConfig = field(default_factory=EKFConfig)
    spatial_memory: SpatialMemoryConfig = field(default_factory=SpatialMemoryConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    arm: ArmConfig = field(default_factory=ArmConfig)
    pickup: PickupConfig = field(default_factory=PickupConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    foxglove: FoxgloveConfig = field(default_factory=FoxgloveConfig)
    phone_imu: PhoneIMUConfig = field(default_factory=PhoneIMUConfig)


def load_config() -> Config:
    cfg = Config()
    if os.environ.get("CHAIR_SERIAL_PORT"):
        cfg.serial.port = os.environ["CHAIR_SERIAL_PORT"]
    if os.environ.get("CHAIR_LIDAR_PORT"):
        cfg.lidar.port = os.environ["CHAIR_LIDAR_PORT"]
    if os.environ.get("VIAM_MACHINE_ADDRESS"):
        cfg.arm.address = os.environ["VIAM_MACHINE_ADDRESS"]
    if os.environ.get("VIAM_API_KEY"):
        cfg.arm.api_key = os.environ["VIAM_API_KEY"]
    if os.environ.get("VIAM_API_KEY_ID"):
        cfg.arm.api_key_id = os.environ["VIAM_API_KEY_ID"]
    if os.environ.get("VIAM_ARM_NAME"):
        cfg.arm.arm_name = os.environ["VIAM_ARM_NAME"]
    if os.environ.get("VIAM_GRIPPER_NAME"):
        cfg.arm.gripper_name = os.environ["VIAM_GRIPPER_NAME"]
    if os.environ.get("VIAM_ARM_TIMEOUT_S"):
        try:
            cfg.arm.default_timeout_s = float(os.environ["VIAM_ARM_TIMEOUT_S"])
        except ValueError:
            pass
    if os.environ.get("VIAM_ARM_FRAME"):
        cfg.arm.default_frame = os.environ["VIAM_ARM_FRAME"]
    cfg.arm.enabled = _env_bool("VIAM_ARM_ENABLED", cfg.arm.enabled)
    cfg.arm.use_fake = _env_bool("VIAM_USE_FAKE_ARM", cfg.arm.use_fake)
    home_positions = _env_float_list("VIAM_ARM_HOME_JOINTS")
    if home_positions:
        cfg.arm.home_joint_positions = home_positions
    return cfg
