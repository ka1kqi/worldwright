"""Typed handles + state snapshots for the worldwright engine wrapper.

All accessors return host numpy arrays — callers never see device tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # wxyz, matches Genesis convention
RGB = tuple[float, float, float]


def to_np(x: Any) -> np.ndarray:
    """Normalize Genesis tensor outputs (torch / MPS / CUDA) to host numpy."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@dataclass(frozen=True)
class ObjectState:
    pos: np.ndarray   # (3,) float
    quat: np.ndarray  # (4,) float, wxyz


@dataclass(frozen=True)
class ContactInfo:
    a_name: str
    b_name: str
    force: np.ndarray     # (3,)
    position: np.ndarray  # (3,)


@dataclass(frozen=True)
class SceneState:
    """Snapshot of the world at one simulation step.

    Reward predicates close over a SceneState, so they are testable in isolation.
    """
    t: float
    franka_q: np.ndarray      # (9,) — 7 arm + 2 fingers
    franka_qdot: np.ndarray   # (9,)
    ee_pos: np.ndarray        # (3,)
    ee_quat: np.ndarray       # (4,) wxyz
    objects: dict[str, ObjectState]
    contacts: list[ContactInfo]


class EntityHandle:
    """Reference to one Genesis entity. Use only the public methods in LLM-generated code."""

    def __init__(self, name: str, entity: Any) -> None:
        self.name = name
        self._entity = entity  # underlying genesis RigidEntity

    def get_pos(self) -> np.ndarray:
        return to_np(self._entity.get_pos())

    def get_quat(self) -> np.ndarray:
        return to_np(self._entity.get_quat())


class FrankaHandle(EntityHandle):
    """Franka Panda — 7 arm DoFs + 2 finger DoFs. End-effector link is ``hand``."""

    ARM_JOINTS: tuple[str, ...] = tuple(f"joint{i}" for i in range(1, 8))
    FINGER_JOINTS: tuple[str, ...] = ("finger_joint1", "finger_joint2")
    END_EFFECTOR_LINK: str = "hand"

    # Tuned for stable PD tracking on the Panda; matches the Genesis tutorial.
    DEFAULT_KP = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100], dtype=float)
    DEFAULT_KV = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10], dtype=float)
    DEFAULT_FORCE_MIN = np.array(
        [-87, -87, -87, -87, -12, -12, -12, -100, -100], dtype=float
    )
    DEFAULT_FORCE_MAX = -DEFAULT_FORCE_MIN

    GRIPPER_OPEN_Q: float = 0.04
    GRASP_QUAT: np.ndarray = np.array([0.0, 1.0, 0.0, 0.0])  # gripper-down

    def __init__(self, name: str, entity: Any) -> None:
        super().__init__(name, entity)
        # Populated by _post_build().
        self.arm_dofs: list[int] = []
        self.finger_dofs: list[int] = []
        self._end_effector: Any = None

    # -- internal --
    def _post_build(self) -> None:
        """Resolve joint indices and apply default control gains. Called by Scene.build()."""
        self.arm_dofs = [
            self._entity.get_joint(n).dofs_idx_local[0] for n in self.ARM_JOINTS
        ]
        self.finger_dofs = [
            self._entity.get_joint(n).dofs_idx_local[0] for n in self.FINGER_JOINTS
        ]
        self._end_effector = self._entity.get_link(self.END_EFFECTOR_LINK)
        self._entity.set_dofs_kp(self.DEFAULT_KP)
        self._entity.set_dofs_kv(self.DEFAULT_KV)
        self._entity.set_dofs_force_range(
            self.DEFAULT_FORCE_MIN, self.DEFAULT_FORCE_MAX
        )

    # -- queries --
    def get_dofs_position(self) -> np.ndarray:
        return to_np(self._entity.get_dofs_position())

    def get_dofs_velocity(self) -> np.ndarray:
        return to_np(self._entity.get_dofs_velocity())

    def get_ee_pos(self) -> np.ndarray:
        return to_np(self._end_effector.get_pos())

    def get_ee_quat(self) -> np.ndarray:
        return to_np(self._end_effector.get_quat())

    # -- planning --
    def ik(self, pos: np.ndarray | Vec3, quat: np.ndarray | Quat | None = None) -> np.ndarray:
        """Solve IK for the end-effector. Returns a 9-vec (arm + fingers)."""
        q = self._entity.inverse_kinematics(
            link=self._end_effector,
            pos=np.asarray(pos, dtype=float),
            quat=np.asarray(quat if quat is not None else self.GRASP_QUAT, dtype=float),
        )
        return to_np(q)

    def plan(self, q_goal: np.ndarray, n_waypoints: int = 200) -> list[np.ndarray]:
        """RRT-Connect motion plan to a 9-vec joint goal."""
        path = self._entity.plan_path(qpos_goal=q_goal, num_waypoints=n_waypoints)
        return [to_np(wp) for wp in path]

    # -- control --
    def move_to(self, q: np.ndarray) -> None:
        """Set PD target for all 9 DoFs."""
        self._entity.control_dofs_position(np.asarray(q, dtype=float))

    def move_arm_to(self, q: np.ndarray) -> None:
        """Set PD target for the 7 arm DoFs only (fingers stay under last command)."""
        self._entity.control_dofs_position(
            np.asarray(q, dtype=float)[:7], self.arm_dofs
        )

    def open_gripper(self) -> None:
        self._entity.control_dofs_position(
            np.array([self.GRIPPER_OPEN_Q, self.GRIPPER_OPEN_Q]), self.finger_dofs
        )

    def close_gripper(self, force: float = -0.5) -> None:
        """Apply finger force (negative = closing). -0.5 is the canonical grasp force."""
        self._entity.control_dofs_force(
            np.array([force, force], dtype=float), self.finger_dofs
        )
