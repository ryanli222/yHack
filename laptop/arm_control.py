"""
Arm control wrapper for VIAM arm + gripper resources.

This module mirrors the repo's other subsystem wrappers:
- async lifecycle methods
- local fake implementation for no-hardware tests
- lazy VIAM imports so the rest of the app can import safely

Environment-backed config lives in config.py under ArmConfig.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence

from config import load_config

logger = logging.getLogger(__name__)


class ArmControlError(RuntimeError):
    """Raised when the arm wrapper cannot complete a requested action."""


@dataclass
class ArmPose:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    o_x: float = 0.0
    o_y: float = 0.0
    o_z: float = 1.0
    theta: float = 0.0

    @classmethod
    def from_value(cls, value: "ArmPose | Mapping[str, Any]") -> "ArmPose":
        if isinstance(value, cls):
            return cls(**asdict(value))
        if not isinstance(value, Mapping):
            raise ArmControlError(f"Invalid pose payload: expected mapping, got {type(value).__name__}")
        try:
            return cls(
                x=float(value.get("x", 0.0)),
                y=float(value.get("y", 0.0)),
                z=float(value.get("z", 0.0)),
                o_x=float(value.get("o_x", 0.0)),
                o_y=float(value.get("o_y", 0.0)),
                o_z=float(value.get("o_z", 1.0)),
                theta=float(value.get("theta", 0.0)),
            )
        except (TypeError, ValueError) as exc:
            raise ArmControlError(f"Invalid pose values: {exc}") from exc

    @classmethod
    def from_proto(cls, pose: Any) -> "ArmPose":
        return cls(
            x=float(getattr(pose, "x", 0.0)),
            y=float(getattr(pose, "y", 0.0)),
            z=float(getattr(pose, "z", 0.0)),
            o_x=float(getattr(pose, "o_x", 0.0)),
            o_y=float(getattr(pose, "o_y", 0.0)),
            o_z=float(getattr(pose, "o_z", 1.0)),
            theta=float(getattr(pose, "theta", 0.0)),
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class ArmState:
    connected: bool = False
    arm_name: str = ""
    gripper_name: str = ""
    joint_positions: list[float] = field(default_factory=list)
    pose: ArmPose = field(default_factory=ArmPose)
    gripper_open: bool = True
    holding: bool = False
    is_moving: bool = False
    last_error: str = ""

    def copy(self) -> "ArmState":
        return ArmState(
            connected=self.connected,
            arm_name=self.arm_name,
            gripper_name=self.gripper_name,
            joint_positions=list(self.joint_positions),
            pose=ArmPose(**self.pose.to_dict()),
            gripper_open=self.gripper_open,
            holding=self.holding,
            is_moving=self.is_moving,
            last_error=self.last_error,
        )


def _normalize_joint_values(values: Sequence[float]) -> list[float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ArmControlError("Joint values must be a sequence of numbers")
    try:
        joints = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ArmControlError(f"Invalid joint values: {exc}") from exc
    if not joints:
        raise ArmControlError("At least one joint value is required")
    return joints


class ArmControl:
    """VIAM-backed arm/gripper wrapper."""

    def __init__(self):
        cfg = load_config().arm
        self._address = cfg.address
        self._api_key = cfg.api_key
        self._api_key_id = cfg.api_key_id
        self._arm_name = cfg.arm_name
        self._gripper_name = cfg.gripper_name
        self._default_timeout_s = cfg.default_timeout_s
        self._home_joint_positions = list(cfg.home_joint_positions)
        self._default_frame = cfg.default_frame

        self._robot = None
        self._arm = None
        self._gripper = None
        self._state = ArmState(
            connected=False,
            arm_name=self._arm_name,
            gripper_name=self._gripper_name,
        )

    async def start(self) -> None:
        await self.connect()

    async def connect(self) -> None:
        if self._state.connected:
            return
        self._validate_connection_config()

        try:
            from viam.robot.client import RobotClient
            from viam.components.arm import Arm
            from viam.components.gripper import Gripper
        except ImportError as exc:
            raise self._remember_and_raise(
                "viam-sdk is not installed; install it before connecting to the arm"
            ) from exc

        robot = None
        try:
            opts = RobotClient.Options.with_api_key(
                api_key=self._api_key,
                api_key_id=self._api_key_id,
            )
            robot = await RobotClient.at_address(self._address, opts)
            self._arm = Arm.from_robot(robot=robot, name=self._arm_name)
            self._gripper = Gripper.from_robot(robot=robot, name=self._gripper_name)
            self._robot = robot
            self._state.connected = True
            self._state.last_error = ""
            await self.get_state()
            logger.info("Connected to VIAM arm '%s' and gripper '%s'", self._arm_name, self._gripper_name)
        except ArmControlError:
            if robot is not None:
                await robot.close()
            raise
        except Exception as exc:
            if robot is not None:
                await robot.close()
            raise self._remember_and_raise(f"Failed to connect to VIAM arm resources: {exc}") from exc

    async def stop(self) -> None:
        if self._arm is not None:
            try:
                await self._arm.stop(timeout=self._default_timeout_s)
            except Exception:
                logger.debug("Ignoring arm stop error during shutdown", exc_info=True)
        if self._gripper is not None:
            try:
                await self._gripper.stop(timeout=self._default_timeout_s)
            except Exception:
                logger.debug("Ignoring gripper stop error during shutdown", exc_info=True)
        if self._robot is not None:
            try:
                await self._robot.close()
            except Exception:
                logger.debug("Ignoring robot close error during shutdown", exc_info=True)
        self._robot = None
        self._arm = None
        self._gripper = None
        self._state.connected = False
        self._state.is_moving = False
        logger.info("Arm control stopped")

    async def home(self, timeout_s: Optional[float] = None) -> None:
        if not self._home_joint_positions:
            raise self._remember_and_raise(
                "Home joint positions are not configured; set VIAM_ARM_HOME_JOINTS for home() support"
            )
        await self.move_joints(self._home_joint_positions, timeout_s=timeout_s)

    async def move_joints(self, values: Sequence[float], timeout_s: Optional[float] = None) -> list[float]:
        joints = _normalize_joint_values(values)
        await self.connect()

        try:
            from viam.proto.component.arm import JointPositions
        except ImportError as exc:
            raise self._remember_and_raise(
                "viam-sdk is not installed; install it before moving arm joints"
            ) from exc

        timeout = timeout_s or self._default_timeout_s
        try:
            self._state.is_moving = True
            await self._arm.move_to_joint_positions(
                positions=JointPositions(values=joints),
                timeout=timeout,
            )
            self._state.joint_positions = joints
            self._state.is_moving = False
            self._state.last_error = ""
            return list(joints)
        except Exception as exc:
            self._state.is_moving = False
            raise self._remember_and_raise(f"Failed to move arm joints: {exc}") from exc

    async def move_to_position(
        self,
        pose: ArmPose | Mapping[str, Any],
        timeout_s: Optional[float] = None,
    ) -> ArmPose:
        pose_value = ArmPose.from_value(pose)
        await self.connect()

        try:
            from viam.proto.common import Pose
        except ImportError as exc:
            raise self._remember_and_raise(
                "viam-sdk is not installed; install it before moving the arm"
            ) from exc

        timeout = timeout_s or self._default_timeout_s
        try:
            self._state.is_moving = True
            await self._arm.move_to_position(
                pose=Pose(**pose_value.to_dict()),
                timeout=timeout,
            )
            self._state.pose = pose_value
            self._state.is_moving = False
            self._state.last_error = ""
            return ArmPose(**pose_value.to_dict())
        except Exception as exc:
            self._state.is_moving = False
            raise self._remember_and_raise(f"Failed to move arm to pose: {exc}") from exc

    async def open_gripper(self, timeout_s: Optional[float] = None) -> None:
        await self.connect()
        timeout = timeout_s or self._default_timeout_s
        try:
            await self._gripper.open(timeout=timeout)
            self._state.gripper_open = True
            self._state.holding = False
            self._state.last_error = ""
        except Exception as exc:
            raise self._remember_and_raise(f"Failed to open gripper: {exc}") from exc

    async def close_gripper(self, timeout_s: Optional[float] = None) -> bool:
        await self.connect()
        timeout = timeout_s or self._default_timeout_s
        try:
            grabbed = await self._gripper.grab(timeout=timeout)
            holding = bool(getattr(grabbed, "is_holding_something", grabbed))
            self._state.gripper_open = False
            self._state.holding = holding
            self._state.last_error = ""
            return holding
        except Exception as exc:
            raise self._remember_and_raise(f"Failed to close gripper: {exc}") from exc

    async def emergency_stop(self) -> None:
        if not self._state.connected:
            return
        try:
            await self._arm.stop(timeout=self._default_timeout_s)
        except Exception:
            logger.debug("Ignoring arm stop error during emergency stop", exc_info=True)
        try:
            await self._gripper.stop(timeout=self._default_timeout_s)
        except Exception:
            logger.debug("Ignoring gripper stop error during emergency stop", exc_info=True)
        self._state.is_moving = False

    async def get_state(self) -> ArmState:
        if not self._state.connected:
            return self._state.copy()
        try:
            joint_positions = await self._arm.get_joint_positions(timeout=self._default_timeout_s)
            pose = await self._arm.get_end_position(timeout=self._default_timeout_s)
            is_moving = await self._arm.is_moving()

            self._state.joint_positions = list(getattr(joint_positions, "values", []))
            self._state.pose = ArmPose.from_proto(pose)
            self._state.is_moving = bool(is_moving)

            try:
                holding_status = await self._gripper.is_holding_something(timeout=self._default_timeout_s)
                self._state.holding = bool(
                    getattr(holding_status, "is_holding_something", holding_status)
                )
            except Exception:
                logger.debug("Could not refresh gripper holding status", exc_info=True)

            self._state.last_error = ""
        except Exception as exc:
            self._state.last_error = f"Failed to refresh arm state: {exc}"
            logger.warning(self._state.last_error)
        return self._state.copy()

    def _validate_connection_config(self) -> None:
        if not self._address:
            raise self._remember_and_raise("VIAM machine address is not configured")
        if not self._api_key or not self._api_key_id:
            raise self._remember_and_raise("VIAM API credentials are not configured")
        if not self._arm_name:
            raise self._remember_and_raise("VIAM arm resource name is not configured")
        if not self._gripper_name:
            raise self._remember_and_raise("VIAM gripper resource name is not configured")

    def _remember_and_raise(self, message: str) -> ArmControlError:
        self._state.last_error = message
        logger.error(message)
        return ArmControlError(message)


class FakeArmControl:
    """No-hardware stub used by unit tests and local demos."""

    def __init__(self, joint_count: int = 5):
        cfg = load_config().arm
        self._joint_count = max(1, joint_count)
        self._state = ArmState(
            connected=False,
            arm_name=cfg.arm_name,
            gripper_name=cfg.gripper_name,
            joint_positions=[0.0] * self._joint_count,
            pose=ArmPose(),
            gripper_open=True,
            holding=False,
            is_moving=False,
        )

    async def start(self) -> None:
        await self.connect()

    async def connect(self) -> None:
        self._state.connected = True
        self._state.last_error = ""
        logger.info("[FAKE] Arm control connected")

    async def stop(self) -> None:
        self._state.connected = False
        self._state.is_moving = False
        logger.info("[FAKE] Arm control stopped")

    async def home(self, timeout_s: Optional[float] = None) -> None:
        del timeout_s
        await self.connect()
        self._state.is_moving = True
        self._state.joint_positions = [0.0] * len(self._state.joint_positions)
        self._state.pose = ArmPose()
        self._state.is_moving = False

    async def move_joints(self, values: Sequence[float], timeout_s: Optional[float] = None) -> list[float]:
        del timeout_s
        joints = _normalize_joint_values(values)
        await self.connect()
        self._state.is_moving = True
        self._state.joint_positions = list(joints)
        self._state.is_moving = False
        return list(self._state.joint_positions)

    async def move_to_position(
        self,
        pose: ArmPose | Mapping[str, Any],
        timeout_s: Optional[float] = None,
    ) -> ArmPose:
        del timeout_s
        pose_value = ArmPose.from_value(pose)
        await self.connect()
        self._state.is_moving = True
        self._state.pose = pose_value
        self._state.is_moving = False
        return ArmPose(**pose_value.to_dict())

    async def open_gripper(self, timeout_s: Optional[float] = None) -> None:
        del timeout_s
        await self.connect()
        self._state.gripper_open = True
        self._state.holding = False

    async def close_gripper(self, timeout_s: Optional[float] = None) -> bool:
        del timeout_s
        await self.connect()
        self._state.gripper_open = False
        self._state.holding = True
        return True

    async def emergency_stop(self) -> None:
        self._state.is_moving = False

    async def get_state(self) -> ArmState:
        return self._state.copy()


def create_arm_control(fake: Optional[bool] = None):
    if fake is None:
        fake = load_config().arm.use_fake
    if fake:
        return FakeArmControl()
    return ArmControl()
