"""
zeeta_robot.py
==============
Zeeta Robot Controller — Franka Panda wrapper.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pybullet as pb

from zeeta.zeeta_physics import ZeetaPhysics

logger = logging.getLogger("ZeetaRobot")

PANDA_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
PANDA_EE_LINK = 11
PANDA_FINGER_JOINTS = [9, 10]


class ZeetaRobot:
    """
    Franka Panda robot controller.
    Wraps joint control, gripper, and end-effector state.
    """

    def __init__(self, physics: ZeetaPhysics):
        self.physics    = physics
        self.robot_id   = physics.add_robot()
        self.ee_link    = PANDA_EE_LINK
        self.n_joints   = 7

    def reset_to_home(self):
        """Reset all joints to home configuration."""
        for j, angle in enumerate(PANDA_HOME):
            pb.resetJointState(
                self.robot_id, j, angle,
                physicsClientId=self.physics._client)
        self.set_gripper(1.0)
        self.physics.step(10)

    def set_gripper(self, open_val: float, steps: int = 30):
        """open_val: 1.0=open, 0.0=closed."""
        target = 0.04 * open_val
        for fj in PANDA_FINGER_JOINTS:
            pb.setJointMotorControl2(
                self.robot_id, fj,
                pb.POSITION_CONTROL,
                targetPosition=target,
                force=20,
                physicsClientId=self.physics._client)
        self.physics.step(steps)

    def move_to(
        self,
        target_pos: Tuple,
        target_orn: Optional[Tuple] = None,
        steps: int = 80,
        carry_body: Optional[int] = None,
    ) -> List[float]:
        """Move end-effector to target position using IK."""
        if target_orn is None:
            target_orn = pb.getQuaternionFromEuler(
                [0, -np.pi, 0],
                physicsClientId=self.physics._client)

        joint_poses = self.physics.calculate_ik(
            self.robot_id, self.ee_link,
            target_pos, target_orn)

        for _ in range(steps):
            self.physics.set_joint_targets_batch(
                self.robot_id,
                {j: float(joint_poses[j]) for j in range(self.n_joints)})
            self.physics.step(1)

            if carry_body is not None:
                ee_pos, _ = self.physics.get_link_state(
                    self.robot_id, self.ee_link)
                carry_pos = [
                    ee_pos[0],
                    ee_pos[1],
                    ee_pos[2] - 0.05]
                self.physics.set_body_pose(carry_body, carry_pos)
                # Zero velocity every step to prevent accumulation
                self.physics._pb.resetBaseVelocity(
                    carry_body, [0, 0, 0], [0, 0, 0],
                    physicsClientId=self.physics._client)

        return self.physics.get_joint_states(self.robot_id)

    def get_ee_pose(self) -> Tuple:
        """Get end-effector position and orientation."""
        return self.physics.get_link_state(
            self.robot_id, self.ee_link)

    def get_joint_states(self) -> List[float]:
        return self.physics.get_joint_states(self.robot_id)

    def __repr__(self):
        return f"ZeetaRobot(id={self.robot_id}, ee_link={self.ee_link})"