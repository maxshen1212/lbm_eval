#! /usr/bin/env python3
"""
LeRobot Multi-Task DiT policy gRPC server for lbm_eval.

Maps ``MultiarmObservation`` to LeRobot-style tensor batches with a sampled
language instruction, runs ``MultiTaskDiTPolicy`` inference, and maps the
predicted action vector back to ``PosesAndGrippers`` for the simulator / OSC stack.

Coordinate frame
----------------
All proprioception and action ``xyz`` / TRI ``rot6d`` components use the **task frame**
(simulation / ``robot.actual.poses``), matching TRI ``observations.npz`` /
``actions.npz`` as copied by ``convert_tri_to_lerobot.py``. No camera (ego) rigid
transform is applied to state or actions.

Tensor / array conventions
----------------------------
* Image tensors: ``float32``, shape ``(B, 3, H, W)``, channel order RGB, values in ``[0, 1]``
  (uint8 inputs scaled by ``1/255``).
* Proprioception ``observation.state``: ``float32``, shape ``(B, 20)``.
* Policy ``action``: ``float32``, length ``20`` after batch/time squeeze.

Rotation parameterization
-------------------------
End-effector orientation uses the **TRI 6D rotation** encoding (first two **rows**
of the ``3×3`` rotation matrix flattened to ``ℝ⁶``), via ``matrix_to_rotation_6d`` /
``rotation_6d_to_matrix``.

Language conditioning
---------------------
On every ``reset_batch``, an instruction is sampled uniformly from the union of
``original`` and ``randomized`` strings for the configured ``task_name`` in
``LBM_data/language_annotations.yaml``. The chosen string is injected into the
observation under the ``"task"`` key on every ``step_batch`` call; the
``TokenizerProcessorStep`` from ``make_multi_task_dit_pre_post_processors``
tokenizes it for the CLIP text encoder.
"""

from __future__ import annotations

import argparse
import copy
import logging
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from pydrake.math import RotationMatrix

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import (
    MultiTaskDiTPolicy,
)
import lerobot.policies.pretrained as _lerobot_pretrained
from safetensors.torch import load_file as _safetensors_load_file


def _load_model_skip_shared_check(model, filename, strict=True, device="cpu"):
    # Bypass safetensors.torch.load_model's _remove_duplicate_names check, which
    # fails when the freshly-instantiated model has shared tensor storage (e.g.
    # CLIPTextModel from clip-vit-base-patch32). The saved checkpoint is fine;
    # the issue is purely in the in-memory model graph.
    state_dict = _safetensors_load_file(filename, device=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing, unexpected = list(missing), list(unexpected)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Error(s) in loading state_dict for {model.__class__.__name__}:"
            f"\n    Missing key(s): {missing}"
            f"\n    Unexpected key(s): {unexpected}"
        )
    return missing, unexpected


_lerobot_pretrained.load_model_as_safetensor = _load_model_skip_shared_check

from robot_gym.multiarm_spaces import MultiarmObservation, PosesAndGrippers
from robot_gym.multiarm_spaces_conversions import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from robot_gym.policy import Policy, PolicyMetadata

from grpc_workspace.lbm_policy_server import (
    LbmPolicyServerConfig,
    run_policy_server,
)

_LOG = logging.getLogger(__name__)

# Default semantic camera key for ``observation.images.*``; overridden by --cameras.
_DEFAULT_CAMERAS = ["scene_right_0"]

# TRI 6D encoding of R = I₃; shape (6,), dtype float64; copied when an arm pose is missing.
_IDENTITY_ROT6D_TRI = matrix_to_rotation_6d(np.eye(3))

_DEFAULT_LANG_YAML = Path("/data/maxshen/LBM_data/language_annotations.yaml")


def _proprio_20d_numpy_from_observation(
    lbm_obs: MultiarmObservation,
) -> np.ndarray:
    """Construct the 20-dimensional proprioceptive state vector (NumPy, no batch dim).

    Layout (interleaved per arm; matches ``EGO_STATE_KEYS`` in ``convert_tri_to_lerobot``):

        Index   Length   Content
        [0:3)      3     Right end-effector position in T, ``xyz`` (meters)
        [3:9)      6     Right end-effector TRI rot6d
        [9:10)     1     Right gripper scalar
        [10:13)    3     Left end-effector position in T, ``xyz``
        [13:19)    6     Left end-effector TRI rot6d
        [19:20)    1     Left gripper scalar
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
        [
            right_xyz,
            right_rot6d,
            right_gripper,
            left_xyz,
            left_rot6d,
            left_gripper,
        ]
    ).astype(np.float32)


def _update_arm_pose_from_task_xyz_rot6d(
    poses: dict,
    arm_key: str,
    xyz: np.ndarray,
    rot6d: np.ndarray,
) -> None:
    """Write task-frame pose from absolute ``xyz`` and TRI ``rot6d``."""
    if arm_key not in poses:
        return
    R_TE = rotation_6d_to_matrix(rot6d.astype(np.float64))
    poses[arm_key].set_translation(xyz)
    poses[arm_key].set_rotation(RotationMatrix(R_TE))


def _load_task_instructions(yaml_path: Path, task_name: str) -> List[str]:
    """Load instruction candidates for ``task_name`` from the language YAML.

    Returns the union of ``original`` and ``randomized`` lists.
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    lang_dict = data.get("language_dict", {})
    if task_name not in lang_dict:
        available = sorted(lang_dict.keys())
        raise KeyError(
            f"Task '{task_name}' not found in {yaml_path}. "
            f"First few available: {available[:10]}..."
        )

    entry = lang_dict[task_name] or {}
    originals = list(entry.get("original") or [])
    randomized = list(entry.get("randomized") or [])
    candidates = originals + randomized
    if not candidates:
        raise ValueError(
            f"Task '{task_name}' in {yaml_path} has no instructions."
        )
    return candidates


class LerobotMultiTaskDiTWrapper(Policy):
    """Batch ``Policy`` implementation wrapping a ``MultiTaskDiTPolicy`` checkpoint
    with language conditioning sourced from a YAML annotation file.

    ``step_batch`` consumes one ``MultiarmObservation`` per environment UUID and
    returns ``PosesAndGrippers`` in the same keying.
    """

    def __init__(
        self,
        model_id: str,
        task_name: str,
        language_annotations_path: Path,
        cameras: List[str],
        device: str = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.model_id = model_id
        self.task_name = task_name
        self.cameras = list(cameras)

        _LOG.info(
            "Loading Multi-Task DiT policy from %s to %s...", model_id, device
        )
        self.policy = MultiTaskDiTPolicy.from_pretrained(model_id).to(
            self.device
        )
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

        self._instructions = _load_task_instructions(
            language_annotations_path, task_name
        )
        _LOG.info(
            "Loaded %d instructions for task '%s'.",
            len(self._instructions),
            task_name,
        )
        # Per-env sampled instruction; replaced on each ``reset_batch``.
        self._env_instruction: Dict[uuid.UUID, str] = {}
        # Used for envs that never went through reset_batch on this server.
        self._fallback_instruction: str = random.choice(self._instructions)
        _LOG.info("Model and processors loaded successfully.")

    def get_policy_metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            name="LeRobot_MultiTaskDiTPolicy",
            skill_type="MultiTaskDiT",
            checkpoint_path=self.model_id,
            git_repo="lerobot",
            git_sha="Undefined",
        )

    def reset_batch(
        self,
        seeds: Optional[Dict[uuid.UUID, int]] = None,
        options: Optional[Dict[uuid.UUID, Dict[str, Any]]] = None,
    ):
        env_uuids = list(seeds.keys()) if seeds else []
        for uid in env_uuids:
            instruction = random.choice(self._instructions)
            self._env_instruction[uid] = instruction
            self._fallback_instruction = instruction
            _LOG.info("Env %s reset with instruction: %r", uid, instruction)
        self.policy.reset()

    def step_batch(
        self, observations: Dict[uuid.UUID, MultiarmObservation]
    ) -> Dict[uuid.UUID, PosesAndGrippers]:
        out: Dict[uuid.UUID, PosesAndGrippers] = {}

        for uid, lbm_obs in observations.items():
            raw_observation = self._build_observation_tensors(lbm_obs)

            instruction = self._env_instruction.get(
                uid, self._fallback_instruction
            )
            raw_observation["task"] = instruction
            _LOG.debug("Env %s using instruction: %r", uid, instruction)

            for cam in self.cameras:
                img_key = f"observation.images.{cam}"
                img_raw = raw_observation[img_key]
                _LOG.debug(
                    "Raw observation %s: shape=%s dtype=%s min=%.6f max=%.6f mean=%.6f",
                    img_key,
                    tuple(img_raw.shape),
                    img_raw.dtype,
                    img_raw.min().item(),
                    img_raw.max().item(),
                    img_raw.mean().item(),
                )
            st_raw = raw_observation["observation.state"]
            _LOG.debug(
                "Raw observation.state: shape=%s dtype=%s min=%.6f max=%.6f mean=%.6f",
                tuple(st_raw.shape),
                st_raw.dtype,
                st_raw.min().item(),
                st_raw.max().item(),
                st_raw.mean().item(),
            )

            normalized_observation = self.pre_processor(raw_observation)

            for key, tensor in normalized_observation.items():
                if isinstance(tensor, torch.Tensor):
                    normalized_observation[key] = tensor.to(self.device)

            with torch.no_grad():
                normalized_action = self.policy.select_action(
                    normalized_observation
                )

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

            out[uid] = self._action_tensor_to_poses_and_grippers(
                unnormalized_action, lbm_obs
            )
        return out

    def _build_observation_tensors(
        self, lbm_obs: MultiarmObservation
    ) -> Dict[str, torch.Tensor]:
        """Assemble the LeRobot observation dict for a single environment.

        Returns a mapping with:

        * ``observation.images.{cam}`` for each cam in ``self.cameras``:
          ``torch.float32``, shape ``(1, 3, H, W)``, RGB, values in ``[0, 1]``.
        * ``observation.state``: ``torch.float32``, shape ``(1, 20)``.
        """
        obs: Dict[str, torch.Tensor] = {}
        for cam in self.cameras:
            if cam not in lbm_obs.visuo:
                raise ValueError(
                    f"Camera '{cam}' not found. Available: {list(lbm_obs.visuo.keys())}"
                )
            img = np.array(lbm_obs.visuo[cam].rgb.array[..., :3], copy=True)
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            obs[f"observation.images.{cam}"] = img_t.unsqueeze(0).to(
                self.device
            )

        state = _proprio_20d_numpy_from_observation(lbm_obs)
        obs["observation.state"] = (
            torch.tensor(state, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )
        return obs

    def _action_tensor_to_poses_and_grippers(
        self, action: torch.Tensor, observation: MultiarmObservation
    ) -> PosesAndGrippers:
        """Decode a 20D policy output into simulator task-frame poses and grippers.

        Action layout (TRI / ``TRAINING_DATA_FORMAT.md``; *not* interleaved):

            Index   Length   Content
            [0:3)      3     Right end-effector position in T, ``xyz``
            [3:9)      6     Right end-effector TRI rot6d in T
            [9:12)     3     Left end-effector position in T, ``xyz``
            [12:18)    6     Left end-effector TRI rot6d in T
            [18]       1     Right gripper command
            [19]       1     Left gripper command
        """
        a = action.squeeze().cpu().numpy()
        poses = copy.deepcopy(observation.robot.actual.poses)
        grippers = copy.deepcopy(observation.robot.actual.grippers)
        joint_position = copy.deepcopy(observation.robot.actual.joint_position)

        _update_arm_pose_from_task_xyz_rot6d(
            poses, "right::panda", a[0:3], a[3:9]
        )
        _update_arm_pose_from_task_xyz_rot6d(
            poses, "left::panda", a[9:12], a[12:18]
        )
        grippers["right::panda_hand"] = float(a[18])
        grippers["left::panda_hand"] = float(a[19])

        return PosesAndGrippers(
            poses=poses, grippers=grippers, joint_position=joint_position
        )


def main():
    parser = argparse.ArgumentParser(
        description="LeRobot Multi-Task DiT gRPC policy server for lbm_eval."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="LeRobot / HF checkpoint directory for the Multi-Task DiT policy.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help=(
            "Task key in language_annotations.yaml "
            "(e.g. 'BimanualPlaceAppleFromBowlOnCuttingBoard')."
        ),
    )
    parser.add_argument(
        "--language_annotations_path",
        type=Path,
        default=_DEFAULT_LANG_YAML,
        help=f"Path to language_annotations.yaml (default: {_DEFAULT_LANG_YAML}).",
    )
    parser.add_argument(
        "--instruction_seed",
        type=int,
        default=42,
        help="Optional RNG seed for instruction sampling.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging level for this server (default: INFO).",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=_DEFAULT_CAMERAS,
        choices=(
            "scene_right_0",
            "scene_left_0",
            "wrist_left_minus",
            "wrist_left_plus",
            "wrist_right_minus",
            "wrist_right_plus",
        ),
        help="Camera(s) to use for observation (default: scene_right_0).",
    )
    LbmPolicyServerConfig.add_argparse_arguments(parser)
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s [%(name)s] %(message)s",
        force=True,
    )

    if args.instruction_seed is not None:
        random.seed(args.instruction_seed)

    policy = LerobotMultiTaskDiTWrapper(
        model_id=args.model_id,
        task_name=args.task_name,
        language_annotations_path=args.language_annotations_path,
        cameras=args.cameras,
    )
    run_policy_server(policy, args)


if __name__ == "__main__":
    main()
