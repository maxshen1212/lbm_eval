#! /usr/bin/env python3
"""
LeRobot Diffusion policy gRPC server for lbm_eval.

Maps ``MultiarmObservation`` to LeRobot-style tensor batches, runs ``DiffusionPolicy``
inference, and maps the predicted action vector back to ``PosesAndGrippers`` for the
simulator / OSC stack.

Coordinate frames
-----------------
The following rigid-body frames are used consistently in this module:

    T — Task (simulation / policy parent frame; same frame as ``robot.actual.poses``).
    C — Primary egocentric RGB camera (semantic name ``scene_right_0`` by default).
    E — End-effector (per arm).

``X_TC`` denotes the rigid transform of **camera C with respect to task T** as provided
by ``MultiarmObservation.visuo[...].rgb.X_TC`` (Drake ``RigidTransform``). Composition
follows Drake conventions: ``X_TB = X_TA.multiply(X_AB)`` expresses frame B in T given
A in T and B in A.

Tensor / array conventions
--------------------------
* Image tensors: ``float32``, shape ``(B, 3, H, W)``, channel order RGB, values in ``[0, 1]``
  (uint8 inputs scaled by ``1/255``).
* Proprioception ``observation.state``: ``float32``, shape ``(B, 20)``; layout documented
  under ``_proprio_20d_numpy_from_observation``.
* Policy ``action``: ``float32``, length ``20`` after batch/time squeeze; layout follows
  TRI ``TRAINING_DATA_FORMAT.md`` (*actions* archive), documented under
  ``_action_tensor_to_poses_and_grippers``.

Rotation parameterization
-------------------------
End-effector orientation uses the **TRI 6D rotation** encoding: the first two **rows** of
the ``3×3`` rotation matrix ``R``, flattened to ``ℝ⁶``, via ``matrix_to_rotation_6d`` /
``rotation_6d_to_matrix`` (``robot_gym.multiarm_spaces_conversions``). This is **not**
the Zhou et al. two-**column** 6D representation.

Data alignment
--------------
Layouts match TRI ``observations.npz`` / ``actions.npz`` as ingested by
``convert_tri_to_lerobot.py`` (see ``EGO_STATE_KEYS`` and ``TRAINING_DATA_FORMAT.md``).
"""
from __future__ import annotations

import argparse
import copy
import uuid
from typing import Any, Dict, Optional

import numpy as np
import torch
from pydrake.math import RigidTransform, RotationMatrix

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

from robot_gym.multiarm_spaces import MultiarmObservation, PosesAndGrippers
from robot_gym.multiarm_spaces_conversions import matrix_to_rotation_6d, rotation_6d_to_matrix
from robot_gym.policy import Policy, PolicyMetadata

from grpc_workspace.lbm_policy_server import LbmPolicyServerConfig, run_policy_server

# Semantic camera key for ``observation.images.*``; must match the LeRobot dataset feature name.
_SCENE_RIGHT_CAMERA = "scene_right_0"

# TRI 6D encoding of R = I₃; shape (6,), dtype float64; copied when an arm pose is missing.
_IDENTITY_ROT6D_TRI = matrix_to_rotation_6d(np.eye(3))


def _ee_in_camera_from_task(X_TC: RigidTransform, X_TE: RigidTransform) -> RigidTransform:
    """Express a task-frame end-effector pose in the primary camera frame.

    Given ``X_TE`` (end-effector E relative to task T) and ``X_TC`` (camera C relative
    to T), returns ``X_CE`` such that the same physical pose is represented with the
    camera as the reference frame:

        X_CE = X_TC⁻¹ · X_TE

    In Drake, this is implemented as ``X_TC.InvertAndCompose(X_TE)``.

    Args:
        X_TC: ``RigidTransform``, shape implied by Drake (``4×4`` homogeneous internally);
            camera frame C with respect to task T.
        X_TE: ``RigidTransform``; end-effector E with respect to task T.

    Returns:
        ``RigidTransform`` ``X_CE``; end-effector E with respect to camera C.
    """
    return X_TC.InvertAndCompose(X_TE)


def _task_pose_from_ego_target(X_TC: RigidTransform, X_CE: RigidTransform) -> RigidTransform:
    """Map an ego-frame Cartesian target to the task frame for control.

    The policy predicts targets in the **camera (ego) frame**. The simulator expects
    ``robot.actual.poses`` in the **task frame**. The mapping is:

        X_TE = X_TC · X_CE

    where ``X_CE`` is the predicted end-effector pose expressed in C.

    Args:
        X_TC: ``RigidTransform``; camera C with respect to task T (same as observation path).
        X_CE: ``RigidTransform``; predicted end-effector pose in camera frame C.

    Returns:
        ``RigidTransform`` ``X_TE``; corresponding end-effector pose in task frame T.
    """
    return X_TC.multiply(X_CE)


def _proprio_20d_numpy_from_observation(
    lbm_obs: MultiarmObservation, camera_name: str = _SCENE_RIGHT_CAMERA
) -> np.ndarray:
    """Construct the 20-dimensional proprioceptive state vector (NumPy, no batch dimension).

    End-effector positions and TRI 6D rotations are computed **in the camera frame** using
    the same ``X_TC`` as the RGB observation, via ``_ee_in_camera_from_task``. Gripper
    scalars are taken from ``robot.actual.grippers`` (width / openness, one scalar per arm).

    **Layout (interleaved per arm; matches ``EGO_STATE_KEYS`` in ``convert_tri_to_lerobot``).**
    Concatenation order along axis 0 yields a single ``(20,)`` vector:

        Index   Length   Content
        [0:3)      3     Right end-effector position in C, ``xyz`` (meters), ``float32``
        [3:9)      6     Right end-effector TRI rot6d, ``float32``
        [9:10)     1     Right gripper scalar, ``float32``
        [10:13)    3     Left end-effector position in C, ``xyz``
        [13:19)    6     Left end-effector TRI rot6d
        [19:20)    1     Left gripper scalar

    If an arm is absent from ``actual.poses``, position defaults to ``0₃`` and rotation
    to a copied **identity** TRI 6D vector.

    Args:
        lbm_obs: Incoming observation; must contain ``visuo[camera_name]``.
        camera_name: Semantic camera key; default ``_SCENE_RIGHT_CAMERA``.

    Returns:
        ``np.ndarray`` of shape ``(20,)``, dtype ``float32``.
    """
    if camera_name not in lbm_obs.visuo:
        raise ValueError(f"Camera '{camera_name}' not found.")
    X_TC = lbm_obs.visuo[camera_name].rgb.X_TC
    actual = lbm_obs.robot.actual

    right_pose = actual.poses.get("right::panda")
    if right_pose:
        xr = _ee_in_camera_from_task(X_TC, right_pose)
        right_xyz, right_rot6d = xr.translation(), matrix_to_rotation_6d(xr.rotation().matrix())
    else:
        right_xyz, right_rot6d = np.zeros(3), _IDENTITY_ROT6D_TRI.copy()
    right_gripper = np.array([actual.grippers.get("right::panda_hand", 0.0)])

    left_pose = actual.poses.get("left::panda")
    if left_pose:
        xl = _ee_in_camera_from_task(X_TC, left_pose)
        left_xyz, left_rot6d = xl.translation(), matrix_to_rotation_6d(xl.rotation().matrix())
    else:
        left_xyz, left_rot6d = np.zeros(3), _IDENTITY_ROT6D_TRI.copy()
    left_gripper = np.array([actual.grippers.get("left::panda_hand", 0.0)])

    return np.concatenate(
        [right_xyz, right_rot6d, right_gripper, left_xyz, left_rot6d, left_gripper]
    ).astype(np.float32)


def _lerobot_policy_metadata(model_id: str) -> PolicyMetadata:
    """Build static metadata advertised to lbm_eval for this checkpoint."""
    return PolicyMetadata(
        name="LeRobot_DiffusionPolicy",
        skill_type="Diffusion",
        checkpoint_path=model_id,
        git_repo="lerobot",
        git_sha="Undefined",
    )


def _update_arm_pose_from_ego_xyz_rot6d(
    poses: dict,
    arm_key: str,
    xyz: np.ndarray,
    rot6d: np.ndarray,
    X_TC: RigidTransform,
) -> None:
    """Write task-frame pose for one arm from an ego-frame position + TRI rot6d target.

    The policy outputs ``xyz`` and ``rot6d`` **in the camera frame** (consistent with
    training targets in ``actions.npz``). This routine decodes orientation to ``R_CE ∈ ℝ^{3×3}``,
    forms ``X_CE = (R_CE, xyz)``, maps to ``X_TE`` via ``_task_pose_from_ego_target``, and
    mutates ``poses[arm_key]`` in place.

    Args:
        poses: Mutable map of arm keys to Drake ``RigidTransform`` task-frame poses.
        arm_key: Dictionary key, e.g. ``"right::panda"``.
        xyz: Shape ``(3,)``; position of E in camera frame C (meters).
        rot6d: Shape ``(6,)``; TRI 6D rotation for E in C (decoded with ``rotation_6d_to_matrix``).
        X_TC: Camera-from-task transform for the same RGB stream used at inference.

    Returns:
        None (updates ``poses[arm_key]`` if present).
    """
    if arm_key not in poses:
        return
    R_CE = rotation_6d_to_matrix(rot6d.astype(np.float64))
    X_CE = RigidTransform(R=RotationMatrix(R_CE), p=xyz)
    X_TE = _task_pose_from_ego_target(X_TC, X_CE)
    poses[arm_key].set_translation(X_TE.translation())
    poses[arm_key].set_rotation(X_TE.rotation())


class LerobotDiffusionWrapper(Policy):
    """Batch ``Policy`` implementation wrapping a Hugging Face ``DiffusionPolicy`` checkpoint.

    ``step_batch`` consumes one ``MultiarmObservation`` per environment UUID and returns
    ``PosesAndGrippers`` in the same keying. Internal batch size for the neural network is
    one environment per forward pass in the current implementation.
    """

    def __init__(self, model_id: str, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.model_id = model_id
        print(f"[LeRobot Server] Loading Diffusion Policy from {model_id} to {device}...")
        self.policy = DiffusionPolicy.from_pretrained(model_id).to(self.device)
        self.policy.eval()
        print("[LeRobot Server] Model loaded successfully.")

    def get_policy_metadata(self) -> PolicyMetadata:
        return _lerobot_policy_metadata(self.model_id)

    def reset_batch(
        self,
        seeds: Optional[Dict[uuid.UUID, int]] = None,
        options: Optional[Dict[uuid.UUID, Dict[str, Any]]] = None,
    ):
        env_uuids = list(seeds.keys()) if seeds else "Unknown"
        print(f"[LeRobot Server] Resetting policy queues for envs: {env_uuids}")
        self.policy.reset()

    def step_batch(
        self, observations: Dict[uuid.UUID, MultiarmObservation]
    ) -> Dict[uuid.UUID, PosesAndGrippers]:
        out: Dict[uuid.UUID, PosesAndGrippers] = {}
        for uid, lbm_obs in observations.items():
            tensors = self._build_observation_tensors(lbm_obs)
            with torch.no_grad():
                action = self.policy.select_action(tensors)
            out[uid] = self._action_tensor_to_poses_and_grippers(action, lbm_obs)
        return out

    def _build_observation_tensors(self, lbm_obs: MultiarmObservation) -> Dict[str, torch.Tensor]:
        """Assemble the LeRobot observation dict for a single environment.

        Returns:
            Mapping with two keys:

            * ``observation.images.{_SCENE_RIGHT_CAMERA}``: ``torch.float32``, shape
              ``(1, 3, H, W)``. Source array is ``HWC`` uint8 RGB; last channel slice ``[..., :3]``;
              layout becomes ``CHW`` via ``permute(2, 0, 1)``; values scaled by ``1/255``.

            * ``observation.state``: ``torch.float32``, shape ``(1, 20)``; row vector is
              ``_proprio_20d_numpy_from_observation(lbm_obs)``.
        """
        if _SCENE_RIGHT_CAMERA not in lbm_obs.visuo:
            raise ValueError(f"Camera '{_SCENE_RIGHT_CAMERA}' not found.")
        img = lbm_obs.visuo[_SCENE_RIGHT_CAMERA].rgb.array[..., :3]
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = img_t.unsqueeze(0)
        state = _proprio_20d_numpy_from_observation(lbm_obs)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return {
            f"observation.images.{_SCENE_RIGHT_CAMERA}": img_t.to(self.device),
            "observation.state": state_t.to(self.device),
        }

    def _action_tensor_to_poses_and_grippers(
        self, action: torch.Tensor, observation: MultiarmObservation
    ) -> PosesAndGrippers:
        """Decode a 20D policy output into simulator task-frame poses and gripper commands.

        **Action layout (TRI / ``TRAINING_DATA_FORMAT.md``; *not* interleaved like ``observation.state``).**
        Let ``a`` be the length-20 NumPy vector after ``squeeze`` and ``cpu().numpy()``:

            Index   Length   Content
            [0:3)      3     Right end-effector position in C, ``xyz`` (meters)
            [3:9)      6     Right end-effector TRI rot6d in C
            [9:12)     3     Left end-effector position in C, ``xyz``
            [12:18)    6     Left end-effector TRI rot6d in C
            [18]       1     Right gripper command (scalar)
            [19]       1     Left gripper command (scalar)

        Task-frame ``RigidTransform`` entries in ``poses`` are deep-copied from the current
        observation, then translation/rotation are overwritten for arms present in the map.
        Gripper dict entries are updated by scalar assignment. Joint positions are copied
        unchanged for pass-through to ``PosesAndGrippers``.

        Args:
            action: Policy tensor; typically shape ``(20,)``, ``(1, 20)``, or ``(1, 1, 20)``
                depending on policy head; leading singleton dimensions are removed by ``squeeze``.
            observation: Source observation (for ``X_TC``, deep copies of ``actual`` fields).

        Returns:
            ``PosesAndGrippers`` with updated ``poses``, ``grippers``, and copied ``joint_position``.
        """
        a = action.squeeze().cpu().numpy()
        poses = copy.deepcopy(observation.robot.actual.poses)
        grippers = copy.deepcopy(observation.robot.actual.grippers)
        joint_position = copy.deepcopy(observation.robot.actual.joint_position)
        X_TC = observation.visuo[_SCENE_RIGHT_CAMERA].rgb.X_TC

        _update_arm_pose_from_ego_xyz_rot6d(poses, "right::panda", a[0:3], a[3:9], X_TC)
        _update_arm_pose_from_ego_xyz_rot6d(poses, "left::panda", a[9:12], a[12:18], X_TC)
        grippers["right::panda_hand"] = float(a[18])
        grippers["left::panda_hand"] = float(a[19])

        return PosesAndGrippers(poses=poses, grippers=grippers, joint_position=joint_position)


def main():
    parser = argparse.ArgumentParser(description="LeRobot Diffusion gRPC policy server for lbm_eval.")
    parser.add_argument("--model_id", type=str, required=True, help="LeRobot / HF checkpoint directory")
    LbmPolicyServerConfig.add_argparse_arguments(parser)
    args = parser.parse_args()
    policy = LerobotDiffusionWrapper(model_id=args.model_id)
    run_policy_server(policy, args)


if __name__ == "__main__":
    main()
