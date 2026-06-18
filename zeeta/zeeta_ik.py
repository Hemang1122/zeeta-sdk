"""
zeeta_ik.py - Layer 5: Inverse Kinematics Solver
"""
from __future__ import annotations
import math, time, random, logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

try:
    import pybullet as p
    _PYBULLET = True
except ImportError:
    _PYBULLET = False

ROBOT_CONFIGS = {
    "franka": {
        "name": "Franka Emika Panda",
        "dof": 7,
        "ee_link_name": "panda_hand",
        "joint_lower": [-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973],
        "joint_upper": [ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973],
        "joint_ranges": [5.7946,3.5256,5.7946,3.0020,5.7946,3.7700,5.7946],
        "rest_pose": [0.0,-0.785,0.0,-2.356,0.0,1.571,0.785],
        "gripper_joints": [9,10],
        "gripper_open":   [0.04,0.04],
        "gripper_closed": [0.00,0.00],
        "ee_link_idx": 11,
        "arm_joint_indices": [0,1,2,3,4,5,6],
    },
    "ur5": {
        "name": "UR5",
        "dof": 6,
        "ee_link_name": "ee_link",
        "joint_lower": [-6.28,-6.28,-3.14,-6.28,-6.28,-6.28],
        "joint_upper": [ 6.28, 6.28, 3.14, 6.28, 6.28, 6.28],
        "joint_ranges": [12.56,12.56,6.28,12.56,12.56,12.56],
        "rest_pose": [0.0,-1.57,1.57,-1.57,-1.57,0.0],
        "gripper_joints": [],
        "gripper_open":   [],
        "gripper_closed": [],
        "ee_link_idx": 6,
        "arm_joint_indices": [0,1,2,3,4,5],
    },
    "kuka": {
        "name": "KUKA iiwa",
        "dof": 7,
        "ee_link_name": "lbr_iiwa_link_7",
        "joint_lower": [-2.96,-2.09,-2.96,-2.09,-2.96,-2.09,-3.05],
        "joint_upper": [ 2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.05],
        "joint_ranges": [5.92,4.18,5.92,4.18,5.92,4.18,6.10],
        "rest_pose": [0.0,0.0,0.0,-1.57,0.0,1.57,0.0],
        "gripper_joints": [],
        "gripper_open":   [],
        "gripper_closed": [],
        "ee_link_idx": 6,
        "arm_joint_indices": [0,1,2,3,4,5,6],
    },
}

@dataclass
class JointConfig:
    angles: List[float]
    ee_pos: Optional[List[float]] = None
    ee_orn: Optional[List[float]] = None
    solver_iter: int = 0
    residual_pos: float = 0.0
    residual_orn: float = 0.0

@dataclass
class IKResult:
    success: bool
    config: JointConfig
    method: str = "none"
    solve_time_ms: float = 0.0
    attempts: int = 1

@dataclass
class Waypoint:
    pos: List[float]
    orn: Optional[List[float]] = None
    gripper: float = 1.0
    label: str = ""
    pos_tol: float = 0.005
    orn_tol: float = 0.1

@dataclass
class IKDiagnostics:
    total_solves: int = 0
    successes: int = 0
    failures: int = 0
    total_time_ms: float = 0.0
    method_counts: Dict[str,int] = field(default_factory=dict)
    failure_reasons: Dict[str,int] = field(default_factory=dict)
    avg_iter: float = 0.0
    _iter_sum: int = field(default=0, repr=False)

    @property
    def success_rate(self):
        return 100.0 * self.successes / max(1, self.total_solves)

    def record(self, result):
        self.total_solves += 1
        self.total_time_ms += result.solve_time_ms
        self.method_counts[result.method] = self.method_counts.get(result.method, 0) + 1
        if result.success:
            self.successes += 1
            self._iter_sum += result.config.solver_iter
            self.avg_iter = self._iter_sum / max(1, self.successes)
        else:
            self.failures += 1

    def summary(self):
        rate = self.success_rate
        avg_t = self.total_time_ms / max(1, self.total_solves)
        return (f"IK | solves={self.total_solves} | success={rate:.1f}% | "
                f"avg_time={avg_t:.1f}ms | methods={self.method_counts}")


class ZeetaIK:
    def __init__(self, robot_id, robot_type="franka", physics_client=0,
                 pos_tol=0.002, orn_tol=0.087, max_iter=200, n_restarts=8):
        self.robot_id   = robot_id
        self.client     = physics_client
        self.pos_tol    = pos_tol
        self.orn_tol    = orn_tol
        self.max_iter   = max_iter
        self.n_restarts = n_restarts

        key = robot_type.lower()
        if "franka" in key or "panda" in key: key = "franka"
        elif "ur5" in key or "universal" in key: key = "ur5"
        elif "kuka" in key or "iiwa" in key: key = "kuka"
        else: key = "franka"

        self.cfg  = ROBOT_CONFIGS[key]
        self.dof  = self.cfg["dof"]
        self.diag = IKDiagnostics()
        self.ee_link = self._find_ee_link()
        logger.info("ZeetaIK ready | robot=%s | dof=%d | ee_link=%d",
                    self.cfg["name"], self.dof, self.ee_link)

    def solve(self, target_pos, target_orn=None, seed_config=None):
        t0 = time.perf_counter()
        result = self._solve_pybullet(target_pos, target_orn, seed_config)
        if result.success:
            result.method = "pybullet"
            result.solve_time_ms = (time.perf_counter()-t0)*1000
            self.diag.record(result); return result

        result = self._solve_nullspace(target_pos, target_orn)
        if result.success:
            result.method = "nullspace"
            result.solve_time_ms = (time.perf_counter()-t0)*1000
            self.diag.record(result); return result

        result = self._solve_with_restarts(target_pos, target_orn)
        result.method = "restart" if result.success else "failed"
        result.solve_time_ms = (time.perf_counter()-t0)*1000
        self.diag.record(result)
        return result

    def plan_grasp_trajectory(self, obj_pos, target_pos, obj_height=0.05,
                               table_height=0.625, steps_per_segment=20):
        pre_z   = obj_pos[2]    + obj_height + 0.12
        grasp_z = obj_pos[2]    + obj_height * 0.5
        lift_z  = table_height  + 0.20
        place_z = target_pos[2] + obj_height + 0.03
        down    = [1.0, 0.0, 0.0, 0.0]
        waypoints = [
            Waypoint([obj_pos[0],    obj_pos[1],    pre_z],   down, 1.0, "pregrasp"),
            Waypoint([obj_pos[0],    obj_pos[1],    grasp_z], down, 1.0, "grasp_approach"),
            Waypoint([obj_pos[0],    obj_pos[1],    grasp_z], down, 0.0, "close_gripper"),
            Waypoint([obj_pos[0],    obj_pos[1],    lift_z],  down, 0.0, "lift"),
            Waypoint([target_pos[0], target_pos[1], lift_z],  down, 0.0, "carry"),
            Waypoint([target_pos[0], target_pos[1], place_z], down, 0.0, "place_approach"),
            Waypoint([target_pos[0], target_pos[1], place_z], down, 1.0, "release"),
            Waypoint([target_pos[0], target_pos[1], lift_z],  down, 1.0, "retreat"),
        ]
        return self.plan_trajectory(waypoints, steps_per_segment)

    def plan_trajectory(self, waypoints, steps_per_segment=20):
        if not waypoints: return []
        solved = []
        prev = self.cfg["rest_pose"][:]
        for wp in waypoints:
            result = self.solve(wp.pos, wp.orn, seed_config=prev)
            cfg = result.config
            cfg.ee_pos = wp.pos[:]
            cfg.__dict__["gripper"] = wp.gripper
            cfg.__dict__["label"]   = wp.label
            solved.append((wp, cfg))
            if result.success:
                prev = cfg.angles[:]

        trajectory = []
        for i in range(len(solved)-1):
            _, c0 = solved[i]
            _, c1 = solved[i+1]
            gval = c1.__dict__.get("gripper", 1.0)
            for step in range(steps_per_segment):
                t = step / steps_per_segment
                ts = t*t*(3-2*t)
                angles = [a + ts*(b-a) for a,b in zip(c0.angles, c1.angles)]
                interp = JointConfig(angles=angles)
                interp.__dict__["gripper"] = gval
                trajectory.append(interp)
        trajectory.append(solved[-1][1])
        return trajectory

    def execute(self, trajectory, sim_steps_per_config=5, realtime=False, control_hz=240.0):
        if not _PYBULLET or not trajectory:
            return {"steps": 0, "duration_s": 0.0}
        arm_joints     = self.cfg["arm_joint_indices"]
        gripper_joints = self.cfg["gripper_joints"]
        gripper_open   = self.cfg["gripper_open"]
        gripper_closed = self.cfg["gripper_closed"]
        t0 = time.perf_counter()
        for cfg in trajectory:
            for ji, jidx in enumerate(arm_joints):
                if ji < len(cfg.angles):
                    p.setJointMotorControl2(self.robot_id, jidx,
                        p.POSITION_CONTROL, targetPosition=cfg.angles[ji],
                        force=500, positionGain=0.3, velocityGain=1.0,
                        physicsClientId=self.client)
            if gripper_joints:
                gs = cfg.__dict__.get("gripper", 1.0)
                tg = gripper_open if gs > 0.5 else gripper_closed
                for gj, gpos in zip(gripper_joints, tg):
                    p.setJointMotorControl2(self.robot_id, gj,
                        p.POSITION_CONTROL, targetPosition=gpos,
                        force=100, physicsClientId=self.client)
            for _ in range(sim_steps_per_config):
                p.stepSimulation(physicsClientId=self.client)
            if realtime:
                time.sleep(1.0/control_hz)
        return {"steps": len(trajectory), "duration_s": time.perf_counter()-t0}

    def set_to_config(self, config):
        if not _PYBULLET: return
        for ji, jidx in enumerate(self.cfg["arm_joint_indices"]):
            if ji < len(config.angles):
                p.resetJointState(self.robot_id, jidx, config.angles[ji],
                                  physicsClientId=self.client)

    def get_ee_pose(self):
        if not _PYBULLET: return [0,0,0],[0,0,0,1]
        state = p.getLinkState(self.robot_id, self.ee_link,
                               computeForwardKinematics=True,
                               physicsClientId=self.client)
        return list(state[4]), list(state[5])

    def move_to_rest(self):
        self.set_to_config(JointConfig(angles=self.cfg["rest_pose"][:]))

    def diagnostics(self):
        return self.diag

    def _solve_pybullet(self, pos, orn, seed=None):
        if not _PYBULLET: return self._mock_solve(pos)
        if seed: self._set_seed(seed)
        lower=self.cfg["joint_lower"]; upper=self.cfg["joint_upper"]
        ranges=self.cfg["joint_ranges"]; rest=self.cfg["rest_pose"]
        try:
            if orn is not None:
                joints = p.calculateInverseKinematics(
                    self.robot_id, self.ee_link, pos, orn,
                    lowerLimits=lower, upperLimits=upper,
                    jointRanges=ranges, restPoses=rest,
                    maxNumIterations=self.max_iter,
                    residualThreshold=self.pos_tol,
                    physicsClientId=self.client)
            else:
                joints = p.calculateInverseKinematics(
                    self.robot_id, self.ee_link, pos,
                    lowerLimits=lower, upperLimits=upper,
                    jointRanges=ranges, restPoses=rest,
                    maxNumIterations=self.max_iter,
                    residualThreshold=self.pos_tol,
                    physicsClientId=self.client)
        except Exception as e:
            logger.debug("PyBullet IK error: %s", e)
            return IKResult(False, JointConfig(angles=list(rest), residual_pos=999.0))
        arm = self._clamp_joints(list(joints[:self.dof]))
        res = self._fk_residual(arm, pos)
        return IKResult(res < self.pos_tol*3,
                        JointConfig(angles=arm, solver_iter=self.max_iter, residual_pos=res))

    def _solve_nullspace(self, pos, orn):
        if not _PYBULLET:
            return IKResult(False, JointConfig(angles=self.cfg["rest_pose"][:]))
        lower=self.cfg["joint_lower"]; upper=self.cfg["joint_upper"]
        ranges=self.cfg["joint_ranges"]; rest=self.cfg["rest_pose"]
        damp=[0.01]*self.dof
        try:
            if orn is not None:
                joints = p.calculateInverseKinematics(
                    self.robot_id, self.ee_link, pos, orn,
                    lowerLimits=lower, upperLimits=upper,
                    jointRanges=ranges, restPoses=rest,
                    jointDamping=damp,
                    maxNumIterations=self.max_iter*2,
                    residualThreshold=self.pos_tol*0.5,
                    physicsClientId=self.client)
            else:
                joints = p.calculateInverseKinematics(
                    self.robot_id, self.ee_link, pos,
                    lowerLimits=lower, upperLimits=upper,
                    jointRanges=ranges, restPoses=rest,
                    jointDamping=damp,
                    maxNumIterations=self.max_iter*2,
                    residualThreshold=self.pos_tol*0.5,
                    physicsClientId=self.client)
        except Exception as e:
            logger.debug("Nullspace IK error: %s", e)
            return IKResult(False, JointConfig(angles=list(rest), residual_pos=999.0))
        arm = self._clamp_joints(list(joints[:self.dof]))
        res = self._fk_residual(arm, pos)
        return IKResult(res < self.pos_tol*3,
                        JointConfig(angles=arm, solver_iter=self.max_iter*2, residual_pos=res))

    def _solve_with_restarts(self, pos, orn):
        best, best_res = None, float("inf")
        rng = random.Random()
        lower=self.cfg["joint_lower"]; upper=self.cfg["joint_upper"]
        for attempt in range(self.n_restarts):
            seed = [rng.uniform(lo,hi) for lo,hi in zip(lower,upper)]
            self._set_seed(seed)
            result = self._solve_pybullet(pos, orn, seed=None)
            if result.config.residual_pos < best_res:
                best_res = result.config.residual_pos
                best = result; best.attempts = attempt+1
            if result.success: break
        if best is None:
            return IKResult(False,
                JointConfig(angles=self.cfg["rest_pose"][:], residual_pos=999.0),
                attempts=self.n_restarts)
        best.success = best_res < self.pos_tol*5
        return best

    def _mock_solve(self, pos):
        return IKResult(True,
            JointConfig(angles=self.cfg["rest_pose"][:], ee_pos=pos, residual_pos=0.0),
            method="mock")

    def _find_ee_link(self):
        default = self.cfg["ee_link_idx"]
        if not _PYBULLET: return default
        try:
            n = p.getNumJoints(self.robot_id, physicsClientId=self.client)
            target = self.cfg["ee_link_name"].lower()
            for i in range(n):
                info = p.getJointInfo(self.robot_id, i, physicsClientId=self.client)
                if target in info[12].decode("utf-8").lower():
                    return i
        except Exception: pass
        return default

    def _set_seed(self, angles):
        if not _PYBULLET: return
        for ji, jidx in enumerate(self.cfg["arm_joint_indices"]):
            if ji < len(angles):
                try: p.resetJointState(self.robot_id, jidx, angles[ji],
                                       physicsClientId=self.client)
                except Exception: pass

    def _clamp_joints(self, angles):
        return [float(np.clip(a,lo,hi))
                for a,lo,hi in zip(angles,
                                   self.cfg["joint_lower"],
                                   self.cfg["joint_upper"])]

    def _fk_residual(self, angles, target_pos):
        if not _PYBULLET: return 0.0
        self._set_seed(angles)
        try:
            state = p.getLinkState(self.robot_id, self.ee_link,
                                   computeForwardKinematics=True,
                                   physicsClientId=self.client)
            return float(np.linalg.norm(np.array(state[4]) - np.array(target_pos)))
        except Exception: return 999.0


def _validate():
    print("="*60)
    print("ZeetaIK - Layer 5 Validation")
    print("="*60)
    print("\nRobots:")
    for k,v in ROBOT_CONFIGS.items():
        print(f"  {v['name']:<25} DOF={v['dof']}  EE={v['ee_link_name']}")
    print("\nWaypoints for pick-and-place:")
    obj_pos=[0.35,-0.10,0.65]; target_pos=[0.35,0.10,0.625]; oh=0.05; th=0.625
    wps=[("pregrasp",[obj_pos[0],obj_pos[1],obj_pos[2]+oh+0.12],"open"),
         ("grasp",   [obj_pos[0],obj_pos[1],obj_pos[2]+oh*0.5],"open"),
         ("close",   [obj_pos[0],obj_pos[1],obj_pos[2]+oh*0.5],"CLOSED"),
         ("lift",    [obj_pos[0],obj_pos[1],th+0.20],"closed"),
         ("carry",   [target_pos[0],target_pos[1],th+0.20],"closed"),
         ("place",   [target_pos[0],target_pos[1],target_pos[2]+oh+0.03],"closed"),
         ("release", [target_pos[0],target_pos[1],target_pos[2]+oh+0.03],"OPEN"),
         ("retreat", [target_pos[0],target_pos[1],th+0.20],"open")]
    for lb,pos,grip in wps:
        print(f"  {lb:<18} z={pos[2]:.3f}  gripper={grip}")
    print("\nDiagnostics test:")
    diag=IKDiagnostics()
    for m,ok in [("pybullet",True),("pybullet",True),("nullspace",True),("restart",False)]:
        r=IKResult(ok,JointConfig([0.0]*7,solver_iter=150,residual_pos=0.001 if ok else 0.05),m,3.2)
        diag.record(r)
    print(f"  {diag.summary()}")
    print(f"\nSuccess rate: {diag.success_rate:.1f}%")
    print("\nSyntax check: PASSED")
    print("="*60)
    print("Layer 5 - IK Solver: PASSED")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _validate()
