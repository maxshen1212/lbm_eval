#! /usr/bin/env python3
"""
LeRobot Diffusion policy gRPC server for lbm_eval.

Maps ``MultiarmObservation`` to LeRobot-style tensor batches, runs ``DiffusionPolicy``
inference, and maps the predicted action vector back to ``PosesAndGrippers`` for the
simulator / OSC stack.

Coordinate frame
----------------
All proprioception and action ``xyz`` / TRI ``rot6d`` components use the **task frame**
(simulation / ``robot.actual.poses``), matching TRI ``observations.npz`` /
``actions.npz`` as copied by ``convert_tri_to_lerobot.py``. No camera (ego) rigid
transform is applied to state or actions.

RGB still comes from ``visuo[scene_right_0]`` for ``observation.images.*`` only.

Tensor / array conventions
----------------------------
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
import logging
import uuid
from typing import Any, Dict, Optional

import numpy as np
import torch
from pydrake.math import RotationMatrix

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

from robot_gym.multiarm_spaces import MultiarmObservation, PosesAndGrippers
from robot_gym.multiarm_spaces_conversions import matrix_to_rotation_6d, rotation_6d_to_matrix
from robot_gym.policy import Policy, PolicyMetadata

from grpc_workspace.lbm_policy_server import LbmPolicyServerConfig, run_policy_server

_LOG = logging.getLogger(__name__)

# Semantic camera key for ``observation.images.*``; must match the LeRobot dataset feature name.
_SCENE_RIGHT_CAMERA = "scene_right_0"

# TRI 6D encoding of R = I₃; shape (6,), dtype float64; copied when an arm pose is missing.
_IDENTITY_ROT6D_TRI = matrix_to_rotation_6d(np.eye(3))


def _proprio_20d_numpy_from_observation(lbm_obs: MultiarmObservation) -> np.ndarray:
    """Construct the 20-dimensional proprioceptive state vector (NumPy, no batch dimension).

    End-effector positions and TRI 6D rotations are read **in the task frame** from
    ``robot.actual.poses``. Gripper scalars come from ``robot.actual.grippers``.

    **Layout (interleaved per arm; matches ``EGO_STATE_KEYS`` in ``convert_tri_to_lerobot``).**
    Concatenation order along axis 0 yields a single ``(20,)`` vector:

        Index   Length   Content
        [0:3)      3     Right end-effector position in T, ``xyz`` (meters), ``float32``
        [3:9)      6     Right end-effector TRI rot6d, ``float32``
        [9:10)     1     Right gripper scalar, ``float32``
        [10:13)    3     Left end-effector position in T, ``xyz``
        [13:19)    6     Left end-effector TRI rot6d
        [19:20)    1     Left gripper scalar

    If an arm is absent from ``actual.poses``, position defaults to ``0₃`` and rotation
    to a copied **identity** TRI 6D vector.

    Returns:
        ``np.ndarray`` of shape ``(20,)``, dtype ``float32``.
    """
    actual = lbm_obs.robot.actual

    right_pose = actual.poses.get("right::panda")
    if right_pose:
        right_xyz = np.asarray(right_pose.translation(), dtype=np.float64)
        right_rot6d = matrix_to_rotation_6d(right_pose.rotation().matrix())
    else:
        right_xyz, right_rot6d = np.zeros(3), _IDENTITY_ROT6D_TRI.copy()
    right_gripper = np.array([actual.grippers.get("right::panda_hand", 0.0)])

    left_pose = actual.poses.get("left::panda")
    if left_pose:
        left_xyz = np.asarray(left_pose.translation(), dtype=np.float64)
        left_rot6d = matrix_to_rotation_6d(left_pose.rotation().matrix())
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


def _update_arm_pose_from_task_xyz_rot6d(
    poses: dict,
    arm_key: str,
    xyz: np.ndarray,
    rot6d: np.ndarray,
) -> None:
    """Write task-frame pose from absolute ``xyz`` and TRI ``rot6d`` in the task frame."""
    if arm_key not in poses:
        return
    R_TE = rotation_6d_to_matrix(rot6d.astype(np.float64))
    poses[arm_key].set_translation(xyz)
    poses[arm_key].set_rotation(RotationMatrix(R_TE))


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
        _LOG.info("Loading Diffusion Policy from %s to %s...", model_id, device)
        self.policy = DiffusionPolicy.from_pretrained(model_id).to(self.device)
        self.policy.eval()

        device_override = {"device": str(self.device)}
        _LOG.info(
            "Loading policy pre/post processors from checkpoint %s (device=%s)...",
            model_id,
            device_override["device"],
        )
        self.pre_processor, self.post_processor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=model_id,
            preprocessor_overrides={"device_processor": device_override},
            postprocessor_overrides={"device_processor": device_override},
        )
        _LOG.info("Model and processors loaded successfully.")

    def get_policy_metadata(self) -> PolicyMetadata:
        return _lerobot_policy_metadata(self.model_id)

    def reset_batch(
        self,
        seeds: Optional[Dict[uuid.UUID, int]] = None,
        options: Optional[Dict[uuid.UUID, Dict[str, Any]]] = None,
    ):
        env_uuids = list(seeds.keys()) if seeds else "Unknown"
        _LOG.info("Resetting policy queues for envs: %s", env_uuids)
        self.policy.reset()

    def step_batch(
        self, observations: Dict[uuid.UUID, MultiarmObservation]
    ) -> Dict[uuid.UUID, PosesAndGrippers]:
        out: Dict[uuid.UUID, PosesAndGrippers] = {}
        img_key = f"observation.images.{_SCENE_RIGHT_CAMERA}"
        for uid, lbm_obs in observations.items():
            raw_observation = self._build_observation_tensors(lbm_obs)

            img_raw = raw_observation[img_key]
            st_raw = raw_observation["observation.state"]
            _LOG.debug(
                "Raw observation %s: shape=%s dtype=%s min=%.6f max=%.6f mean=%.6f",
                img_key,
                tuple(img_raw.shape),
                img_raw.dtype,
                img_raw.min().item(),
                img_raw.max().item(),
                img_raw.mean().item(),
            )
            _LOG.debug(
                "Raw observation.state: shape=%s dtype=%s min=%.6f max=%.6f mean=%.6f",
                tuple(st_raw.shape),
                st_raw.dtype,
                st_raw.min().item(),
                st_raw.max().item(),
                st_raw.mean().item(),
            )

            normalized_observation = self.pre_processor(raw_observation)

            img_n = normalized_observation.get(img_key)
            st_n = normalized_observation.get("observation.state")
            if isinstance(img_n, torch.Tensor):
                _LOG.debug(
                    "Normalized %s: shape=%s min=%.6f max=%.6f mean=%.6f",
                    img_key,
                    tuple(img_n.shape),
                    img_n.min().item(),
                    img_n.max().item(),
                    img_n.mean().item(),
                )
            else:
                _LOG.debug("Normalized %s: %r", img_key, img_n)
            if isinstance(st_n, torch.Tensor):
                _LOG.debug(
                    "Normalized observation.state: shape=%s min=%.6f max=%.6f mean=%.6f",
                    tuple(st_n.shape),
                    st_n.min().item(),
                    st_n.max().item(),
                    st_n.mean().item(),
                )
            else:
                _LOG.debug("Normalized observation.state: %r", st_n)
            for key, tensor in normalized_observation.items():
                if isinstance(tensor, torch.Tensor):
                    normalized_observation[key] = tensor.to(self.device)

            with torch.no_grad():
                normalized_action = self.policy.select_action(normalized_observation)

            _LOG.debug(
                "Predicted normalized action: shape=%s dtype=%s min=%.6f max=%.6f mean=%.6f",
                tuple(normalized_action.shape),
                normalized_action.dtype,
                normalized_action.min().item(),
                normalized_action.max().item(),
                normalized_action.float().mean().item(),
            )

            unnormalized_action = self.post_processor(normalized_action)
            _LOG.debug(
                "Post-processed action: shape=%s dtype=%s min=%.6f max=%.6f mean=%.6f",
                tuple(unnormalized_action.shape),
                unnormalized_action.dtype,
                unnormalized_action.min().item(),
                unnormalized_action.max().item(),
                unnormalized_action.float().mean().item(),
            )

            out[uid] = self._action_tensor_to_poses_and_grippers(unnormalized_action, lbm_obs)
        return out

    def _build_observation_tensors(self, lbm_obs: MultiarmObservation) -> Dict[str, torch.Tensor]:
        """Assemble the LeRobot observation dict for a single environment.

        Returns:
            Mapping with two keys:

            * ``observation.images.{_SCENE_RIGHT_CAMERA}``: ``torch.float32``, shape
              ``(1, 3, H, W)``. Source array is ``HWC`` uint8 RGB; last channel slice ``[..., :3]``;
              layout becomes ``CHW`` via ``permute(2, 0, 1)``; values scaled by ``1/255``.

            * ``observation.state``: ``torch.float32``, shape ``(1, 20)``; task-frame proprio from
              ``_proprio_20d_numpy_from_observation(lbm_obs)``.
        """
        if _SCENE_RIGHT_CAMERA not in lbm_obs.visuo:
            raise ValueError(f"Camera '{_SCENE_RIGHT_CAMERA}' not found.")
        # Copy: gRPC-backed buffers can be read-only; PyTorch warns on non-writable numpy.
        img = np.array(lbm_obs.visuo[_SCENE_RIGHT_CAMERA].rgb.array[..., :3], copy=True)
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
            [0:3)      3     Right end-effector position in T, ``xyz`` (meters)
            [3:9)      6     Right end-effector TRI rot6d in T
            [9:12)     3     Left end-effector position in T, ``xyz``
            [12:18)    6     Left end-effector TRI rot6d in T
            [18]       1     Right gripper command (scalar)
            [19]       1     Left gripper command (scalar)

        Task-frame ``RigidTransform`` entries in ``poses`` are deep-copied from the current
        observation, then translation/rotation are overwritten for arms present in the map.
        Gripper dict entries are updated by scalar assignment. Joint positions are copied
        unchanged for pass-through to ``PosesAndGrippers``.

        Args:
            action: Policy tensor; typically shape ``(20,)``, ``(1, 20)``, or ``(1, 1, 20)``
                depending on policy head; leading singleton dimensions are removed by ``squeeze``.
            observation: Source observation (for deep copies of ``actual`` fields).

        Returns:
            ``PosesAndGrippers`` with updated ``poses``, ``grippers``, and copied ``joint_position``.
        """
        a = action.squeeze().cpu().numpy()
        poses = copy.deepcopy(observation.robot.actual.poses)
        grippers = copy.deepcopy(observation.robot.actual.grippers)
        joint_position = copy.deepcopy(observation.robot.actual.joint_position)

        _update_arm_pose_from_task_xyz_rot6d(poses, "right::panda", a[0:3], a[3:9])
        _update_arm_pose_from_task_xyz_rot6d(poses, "left::panda", a[9:12], a[12:18])
        grippers["right::panda_hand"] = float(a[18])
        grippers["left::panda_hand"] = float(a[19])

        return PosesAndGrippers(poses=poses, grippers=grippers, joint_position=joint_position)


def main():
    parser = argparse.ArgumentParser(description="LeRobot Diffusion gRPC policy server for lbm_eval.")
    parser.add_argument("--model_id", type=str, required=True, help="LeRobot / HF checkpoint directory")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level for this server (default: INFO). Use DEBUG to print per-step observation stats.",
    )
    LbmPolicyServerConfig.add_argparse_arguments(parser)
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s [%(name)s] %(message)s",
        force=True,
    )

    policy = LerobotDiffusionWrapper(model_id=args.model_id)
    run_policy_server(policy, args)


if __name__ == "__main__":
    main()
