"""
zeeta_physics.py
================
Zeeta Physics Engine — PyBullet wrapper.
Provides the ZeetaPhysics interface used by all other Zeeta modules.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple, List, Dict

import pybullet as pb
import pybullet_data
import numpy as np

logger = logging.getLogger("ZeetaPhysics")


class ZeetaPhysics:
    """
    Thin wrapper around PyBullet.

    Exposes:
      _pb      — the pybullet module (for direct calls in other modules)
      _client  — physics client ID
    """

    def __init__(
        self,
        headless: bool = True,
        timestep: float = 1.0 / 240.0,
        solver_iterations: int = 50,
    ):
        self.headless           = headless
        self.timestep           = timestep
        self.solver_iterations  = solver_iterations
        self._pb                = pb
        self._client: int       = -1
        self._bodies: List[int] = []

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.disconnect()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self):
        if self._client >= 0:
            return
        mode = pb.DIRECT if self.headless else pb.GUI
        self._client = pb.connect(mode)
        pb.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=self._client)
        pb.setTimeStep(self.timestep, physicsClientId=self._client)
        pb.setPhysicsEngineParameter(
            numSolverIterations=self.solver_iterations,
            physicsClientId=self._client)
        logger.info(
            f"ZeetaPhysics connected "
            f"[{'DIRECT' if self.headless else 'GUI'}] "
            f"client={self._client} "
            f"dt={self.timestep*1000:.2f}ms "
            f"solver_iter={self.solver_iterations}")

    def disconnect(self):
        if self._client >= 0:
            try:
                pb.disconnect(physicsClientId=self._client)
            except Exception:
                pass
            self._client = -1
            self._bodies = []
            logger.info("ZeetaPhysics disconnected.")

    # ------------------------------------------------------------------
    # Scene management
    # ------------------------------------------------------------------

    def reset(self):
        """Reset simulation — remove all bodies, restore gravity."""
        pb.resetSimulation(physicsClientId=self._client)
        pb.setGravity(0, 0, -9.81, physicsClientId=self._client)
        pb.setTimeStep(self.timestep, physicsClientId=self._client)
        self._bodies = []

    def add_plane(self, position: Tuple = (0, 0, 0)) -> int:
        """Load ground plane. Returns body ID."""
        body_id = pb.loadURDF(
            "plane.urdf",
            basePosition=position,
            physicsClientId=self._client)
        self._bodies.append(body_id)
        return body_id

    def add_table(self, position: Tuple = (0.5, 0, 0)) -> int:
        """Load standard table. Returns body ID."""
        body_id = pb.loadURDF(
            "table/table.urdf",
            basePosition=position,
            physicsClientId=self._client)
        self._bodies.append(body_id)
        return body_id

    def add_box(
        self,
        position: Tuple = (0.4, 0.0, 0.65),
        size: float = 0.05,
        mass: float = 0.2,
        color: Tuple = (0.8, 0.2, 0.2, 1.0),
    ) -> int:
        """Add a cube object. Returns body ID."""
        body_id = pb.loadURDF(
            "cube_small.urdf",
            basePosition=position,
            globalScaling=size / 0.05,
            physicsClientId=self._client)
        pb.changeDynamics(
            body_id, -1,
            mass=mass,
            lateralFriction=0.6,
            physicsClientId=self._client)
        pb.changeVisualShape(
            body_id, -1,
            rgbaColor=color,
            physicsClientId=self._client)
        self._bodies.append(body_id)
        return body_id

    def add_robot(
        self,
        urdf: str = "franka_panda/panda.urdf",
        position: Tuple = (0, 0, 0),
        fixed: bool = True,
    ) -> int:
        """Load a robot URDF. Returns body ID."""
        try:
            body_id = pb.loadURDF(
                urdf,
                basePosition=position,
                useFixedBase=fixed,
                physicsClientId=self._client)
        except Exception:
            # Fallback to kuka
            body_id = pb.loadURDF(
                "kuka_iiwa/model.urdf",
                basePosition=position,
                useFixedBase=fixed,
                physicsClientId=self._client)
        self._bodies.append(body_id)
        return body_id

    # ------------------------------------------------------------------
    # Simulation stepping
    # ------------------------------------------------------------------

    def step(self, n: int = 1):
        """Step simulation n times."""
        for _ in range(n):
            pb.stepSimulation(physicsClientId=self._client)

    # ------------------------------------------------------------------
    # Joint control
    # ------------------------------------------------------------------

    def set_joint_targets_batch(
        self,
        body_id: int,
        targets: Dict[int, float],
        max_force: float = 240.0,
        max_velocity: float = 0.5,
    ):
        """Set position targets for multiple joints."""
        for joint_idx, angle in targets.items():
            pb.setJointMotorControl2(
                body_id, joint_idx,
                pb.POSITION_CONTROL,
                targetPosition=angle,
                force=max_force,
                maxVelocity=max_velocity,
                physicsClientId=self._client)

    def get_joint_states(
        self,
        body_id: int,
        n_joints: int = 7,
    ) -> List[float]:
        """Get current joint positions."""
        return [
            float(pb.getJointState(
                body_id, j,
                physicsClientId=self._client)[0])
            for j in range(n_joints)
        ]

    def get_link_state(
        self,
        body_id: int,
        link_index: int,
    ) -> Tuple:
        """Get link world position and orientation."""
        state = pb.getLinkState(
            body_id, link_index,
            physicsClientId=self._client)
        return state[4], state[5]  # world pos, world orn

    def get_body_state(
        self,
        body_id: int,
    ) -> Tuple:
        """Get base position and orientation of a body."""
        pos, orn = pb.getBasePositionAndOrientation(
            body_id, physicsClientId=self._client)
        return list(pos), list(orn)

    def set_body_pose(
        self,
        body_id: int,
        position: Tuple,
        orientation: Tuple = (0, 0, 0, 1),
    ):
        """Teleport a body to a given pose."""
        pb.resetBasePositionAndOrientation(
            body_id, position, orientation,
            physicsClientId=self._client)

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def get_camera_image(
        self,
        cam_eye: Tuple,
        cam_target: Tuple,
        width: int = 224,
        height: int = 224,
        fov: float = 60.0,
    ) -> Tuple:
        """
        Render camera image.
        Returns (rgb, depth, seg) as numpy arrays.
        """
        view = pb.computeViewMatrix(
            cam_eye, cam_target, [0, 0, 1],
            physicsClientId=self._client)
        proj = pb.computeProjectionMatrixFOV(
            fov=fov, aspect=width/height,
            nearVal=0.01, farVal=10.0,
            physicsClientId=self._client)

        _, _, rgb, depth, seg = pb.getCameraImage(
            width=width, height=height,
            viewMatrix=view, projectionMatrix=proj,
            renderer=pb.ER_TINY_RENDERER,
            physicsClientId=self._client)

        rgb_arr   = np.array(rgb,   dtype=np.uint8).reshape(height, width, 4)[:,:,:3]
        depth_arr = np.array(depth, dtype=np.float32).reshape(height, width)
        seg_arr   = np.array(seg,   dtype=np.int32).reshape(height, width)

        near, far = 0.01, 10.0
        depth_real = far * near / (far - (far - near) * depth_arr)

        return rgb_arr, depth_real, seg_arr

    # ------------------------------------------------------------------
    # IK
    # ------------------------------------------------------------------

    def calculate_ik(
        self,
        body_id: int,
        ee_link: int,
        target_pos: Tuple,
        target_orn: Optional[Tuple] = None,
        max_iter: int = 300,
    ) -> np.ndarray:
        """Compute IK solution using PyBullet."""
        kwargs = dict(
            bodyUniqueId=body_id,
            endEffectorLinkIndex=ee_link,
            targetPosition=target_pos,
            maxNumIterations=max_iter,
            residualThreshold=1e-5,
            physicsClientId=self._client,
        )
        if target_orn is not None:
            kwargs["targetOrientation"] = target_orn
        return np.array(
            pb.calculateInverseKinematics(**kwargs),
            dtype=np.float64)[:7]

    def __repr__(self):
        status = "connected" if self._client >= 0 else "disconnected"
        return f"ZeetaPhysics({status}, client={self._client})"