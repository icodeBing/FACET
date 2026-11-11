"""
This script is used to play and visualize a policy in the environment.
"""

import time
import torch
import hydra
import itertools
import datetime
import copy
import csv
import logging
from collections import deque
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf, DictConfig

from torchrl.envs.utils import set_exploration_type, ExplorationType

import active_adaptation as aa
from active_adaptation.utils.export import export_onnx
from active_adaptation.utils.timerfd import Timer
from active_adaptation.utils.helpers import EpisodeStats
from active_adaptation.learning.modules.vecnorm import VecNorm
from active_adaptation.utils.wandb import parse_checkpoint
from active_adaptation.utils.math import (
    euler_from_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate,
    quat_rotate_inverse,
    wrap_to_pi,
    yaw_quat,
)

FILE_PATH = Path(__file__).parent
CONFIG_PATH = FILE_PATH.parent / "cfg"


class GaitCsvRecorder:
    """Write single-environment G1 gait state at every policy control step."""

    def __init__(self, env, path: Path, task_cfg):
        self.env = getattr(env, "base_env", env)
        if self.env.num_envs != 1:
            raise ValueError("record_gait_csv requires task.num_envs=1")
        if self.env.backend != "isaac":
            raise ValueError("record_gait_csv currently requires backend=isaac")

        self.asset = self.env.scene.articulations["robot"]
        self.contact_sensor = self.env.scene.sensors["contact_forces"]
        foot_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
        knee_names = ["left_knee_joint", "right_knee_joint"]
        hip_names = ["left_hip_pitch_joint", "right_hip_pitch_joint"]
        leg_joint_names = [
            "left_hip_pitch_joint", "right_hip_pitch_joint",
            "left_hip_roll_joint", "right_hip_roll_joint",
            "left_hip_yaw_joint", "right_hip_yaw_joint",
            "left_knee_joint", "right_knee_joint",
            "left_ankle_pitch_joint", "right_ankle_pitch_joint",
            "left_ankle_roll_joint", "right_ankle_roll_joint",
        ]
        self.foot_ids = self.asset.find_bodies(foot_names, preserve_order=True)[0]
        self.contact_ids = self.contact_sensor.find_bodies(
            foot_names, preserve_order=True
        )[0]
        self.knee_ids = self.asset.find_joints(knee_names, preserve_order=True)[0]
        self.hip_ids = self.asset.find_joints(hip_names, preserve_order=True)[0]
        self.leg_joint_ids = self.asset.find_joints(
            leg_joint_names, preserve_order=True
        )[0]
        self.torso_id = self.asset.find_bodies(["torso_link"], preserve_order=True)[0]
        if (
            len(self.foot_ids) != 2
            or len(self.contact_ids) != 2
            or len(self.knee_ids) != 2
            or len(self.hip_ids) != 2
            or len(self.leg_joint_ids) != len(leg_joint_names)
            or len(self.torso_id) != 1
        ):
            raise ValueError("G1 gait telemetry could not resolve both feet and knees")
        self.torso_id = self.torso_id[0]

        self.period = float(task_cfg.observation.policy.gait_phase.period)
        self.stance_ratio = float(
            task_cfg.reward.loco.biped_alternating_contact.threshold
        )
        self.phase_offset = getattr(self.env.command_manager, "gait_phase_offset", None)
        self.prev_contact = None
        self.prev_episode_id = None
        self.last_touchdown_world = [None, None]
        self.last_touchdown_time = [None, None]

        try:
            import trimesh
            import rtree  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "record_gait_csv requires trimesh and rtree from the IsaacLab environment"
            ) from exc

        logging.getLogger("trimesh").setLevel(logging.WARNING)
        self.trimesh = trimesh
        mesh_dir = (
            FILE_PATH.parent
            / "active_adaptation"
            / "assets"
            / "G1"
            / "waist_unlocked_urdf"
            / "meshes"
        )
        mesh_paths = [
            mesh_dir / "left_ankle_roll_link_collisions_collision.stl",
            mesh_dir / "right_ankle_roll_link_collisions_collision.stl",
        ]
        missing_meshes = [str(path) for path in mesh_paths if not path.is_file()]
        if missing_meshes:
            raise FileNotFoundError(
                "Training-asset foot collision meshes not found: "
                + ", ".join(missing_meshes)
            )
        self.local_meshes = [
            trimesh.load_mesh(str(mesh_path), process=True).convex_hull
            for mesh_path in mesh_paths
        ]
        self.local_vertices = [
            torch.as_tensor(mesh.vertices.copy(), dtype=torch.float32, device=self.env.device)
            for mesh in self.local_meshes
        ]
        self.local_surface_samples = []
        for mesh in self.local_meshes:
            edge_midpoints = mesh.vertices[mesh.edges_unique].mean(axis=1)
            samples = np.concatenate((mesh.vertices, edge_midpoints, mesh.triangles_center))
            sample_ids = np.linspace(
                0, samples.shape[0] - 1, num=min(256, samples.shape[0]), dtype=np.int64
            )
            self.local_surface_samples.append(
                torch.as_tensor(
                    samples[sample_ids], dtype=torch.float32, device=self.env.device
                )
            )
        self.local_faces = [mesh.faces.copy() for mesh in self.local_meshes]

        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", newline="", encoding="utf-8")
        self.columns = [
            "control_step", "sim_time_s", "episode_id", "episode_step",
            "transition_done", "gait_phase", "left_phase", "right_phase",
            "left_expected_stance", "right_expected_stance",
            "left_contact", "right_contact", "left_air_time_s", "right_air_time_s",
            "left_foot_x_yaw_m", "left_foot_y_yaw_m", "left_foot_z_m",
            "right_foot_x_yaw_m", "right_foot_y_yaw_m", "right_foot_z_m",
            "signed_lateral_gap_m", "foot_center_distance_m",
            "collision_mesh_sampled_surface_gap_est_m",
            "left_mesh_vertex_inside_right", "right_mesh_vertex_inside_left",
            "left_touchdown", "right_touchdown",
            "left_touchdown_x_yaw_m", "left_touchdown_y_yaw_m",
            "right_touchdown_x_yaw_m", "right_touchdown_y_yaw_m",
            "left_step_forward_m", "right_step_forward_m",
            "left_touchdown_interval_s", "right_touchdown_interval_s",
            "left_knee_angle_rad", "right_knee_angle_rad",
            "left_knee_offset_rad", "right_knee_offset_rad",
            "left_hip_pitch_angle_rad", "right_hip_pitch_angle_rad",
            "left_hip_pitch_offset_rad", "right_hip_pitch_offset_rad",
            "left_hip_pitch_vel_radps", "right_hip_pitch_vel_radps",
            "left_hip_roll_vel_radps", "right_hip_roll_vel_radps",
            "left_hip_yaw_vel_radps", "right_hip_yaw_vel_radps",
            "left_knee_vel_radps", "right_knee_vel_radps",
            "left_ankle_pitch_vel_radps", "right_ankle_pitch_vel_radps",
            "left_ankle_roll_vel_radps", "right_ankle_roll_vel_radps",
            "left_hip_pitch_acc_radps2", "right_hip_pitch_acc_radps2",
            "left_hip_roll_acc_radps2", "right_hip_roll_acc_radps2",
            "left_hip_yaw_acc_radps2", "right_hip_yaw_acc_radps2",
            "left_knee_acc_radps2", "right_knee_acc_radps2",
            "left_ankle_pitch_acc_radps2", "right_ankle_pitch_acc_radps2",
            "left_ankle_roll_acc_radps2", "right_ankle_roll_acc_radps2",
            "left_leg_joint_vel_rms_radps", "right_leg_joint_vel_rms_radps",
            "left_leg_joint_acc_rms_radps2", "right_leg_joint_acc_rms_radps2",
            "left_foot_pitch_rel_torso_rad", "right_foot_pitch_rel_torso_rad",
            "left_foot_roll_rel_torso_rad", "right_foot_roll_rel_torso_rad",
            "foot_roll_difference_rad", "foot_roll_mean_rad",
            "left_foot_yaw_error_rad", "right_foot_yaw_error_rad",
            "left_foot_vertical_vel_mps", "right_foot_vertical_vel_mps",
            "root_com_y_m", "root_com_height_m", "root_roll_rad", "root_pitch_rad",
            "command_vx_yaw_mps", "command_vy_yaw_mps",
            "facet_reference_vx_yaw_mps", "facet_reference_vy_yaw_mps",
            "actual_com_vx_yaw_mps", "actual_com_vy_yaw_mps",
            "root_yaw_rad", "command_heading_w_rad", "heading_error_rad",
            "root_roll_rate_radps", "root_pitch_rate_radps", "root_yaw_rate_radps",
        ]
        self.writer = csv.DictWriter(self.file, fieldnames=self.columns)
        self.writer.writeheader()
        print(f"Gait CSV recording to: {path.resolve()}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    @staticmethod
    def _float(value):
        return float(value.detach().cpu().item())

    def _world_collision_mesh(self, foot_index: int, pos_w, quat_w):
        vertices = self.local_vertices[foot_index]
        quat = quat_w.expand(vertices.shape[0], -1)
        world_vertices = quat_rotate(quat, vertices) + pos_w
        samples = self.local_surface_samples[foot_index]
        sample_quat = quat_w.expand(samples.shape[0], -1)
        world_samples = quat_rotate(sample_quat, samples) + pos_w
        mesh = self.trimesh.Trimesh(
            vertices=world_vertices.detach().cpu().numpy(),
            faces=self.local_faces[foot_index],
            process=False,
        )
        return mesh, world_samples.detach().cpu().numpy()

    def capture(self, control_step: int):
        asset = self.asset
        command = self.env.command_manager
        root_pos = asset.data.root_link_pos_w[0]
        root_quat = asset.data.root_link_quat_w[0]
        root_rpy = euler_from_quat(root_quat)
        root_com_pos = asset.data.root_com_pos_w[0]
        yaw_only = yaw_quat(root_quat)
        foot_pos_w = asset.data.body_link_pos_w[0, self.foot_ids]
        foot_pos_b = quat_rotate_inverse(
            yaw_only.expand(2, -1), foot_pos_w - root_pos.unsqueeze(0)
        )
        foot_quat_w = asset.data.body_link_quat_w[0, self.foot_ids]
        foot_lin_vel_w = asset.data.body_lin_vel_w[0, self.foot_ids]
        torso_quat = asset.data.body_link_quat_w[0, self.torso_id]
        foot_relative_torso = quat_mul(
            quat_conjugate(torso_quat).expand(2, -1), foot_quat_w
        )
        foot_relative_rpy = euler_from_quat(foot_relative_torso)
        foot_world_rpy = euler_from_quat(foot_quat_w)
        foot_yaw_error = wrap_to_pi(foot_world_rpy[:, 2] - root_rpy[2])
        contacts = (
            self.contact_sensor.data.current_contact_time[0, self.contact_ids] > 0.0
        )
        episode_id = int(self.env.episode_id[0].item())
        episode_step = int(self.env.episode_length_buf[0].item())
        episode_time = episode_step * float(self.env.step_dt)
        phase_offset = (
            float(self.phase_offset[0].item()) if self.phase_offset is not None else 0.0
        )
        phase = (episode_time / self.period + phase_offset) % 1.0
        foot_phases = [phase, (phase + 0.5) % 1.0]
        expected_stance = [value < self.stance_ratio for value in foot_phases]

        if self.prev_episode_id != episode_id:
            self.prev_contact = contacts.clone()
            self.prev_episode_id = episode_id
            self.last_touchdown_world = [None, None]
            self.last_touchdown_time = [None, None]
            touchdowns = [False, False]
        else:
            touchdowns = [
                bool(contacts[i].item()) and not bool(self.prev_contact[i].item())
                for i in range(2)
            ]
            self.prev_contact = contacts.clone()

        world_meshes, world_samples = zip(*[
            self._world_collision_mesh(i, foot_pos_w[i], foot_quat_w[i])
            for i in range(2)
        ])
        vertex_surface_gaps = [
            self.trimesh.proximity.closest_point(
                world_meshes[1], world_samples[0]
            )[1].min(),
            self.trimesh.proximity.closest_point(
                world_meshes[0], world_samples[1]
            )[1].min(),
        ]
        mesh_gap_est = float(min(vertex_surface_gaps))
        try:
            left_inside = bool(world_meshes[1].contains(world_samples[0]).any())
            right_inside = bool(world_meshes[0].contains(world_samples[1]).any())
        except (ModuleNotFoundError, ValueError):
            left_inside = right_inside = False
        if left_inside or right_inside:
            mesh_gap_est = 0.0

        touchdown_data = []
        for i in range(2):
            if touchdowns[i]:
                previous_pos = self.last_touchdown_world[i]
                previous_time = self.last_touchdown_time[i]
                if previous_pos is None:
                    step_forward = ""
                    interval = ""
                else:
                    displacement_w = foot_pos_w[i] - previous_pos
                    step_forward = self._float(
                        quat_rotate_inverse(yaw_only, displacement_w)[0]
                    )
                    interval = episode_time - previous_time
                touchdown_data.append((step_forward, interval))
                self.last_touchdown_world[i] = foot_pos_w[i].clone()
                self.last_touchdown_time[i] = episode_time
            else:
                touchdown_data.append(("", ""))

        default_knees = asset.data.default_joint_pos[0, self.knee_ids]
        knees = asset.data.joint_pos[0, self.knee_ids]
        hips = asset.data.joint_pos[0, self.hip_ids]
        default_hips = asset.data.default_joint_pos[0, self.hip_ids]
        leg_vel = asset.data.joint_vel[0, self.leg_joint_ids]
        leg_acc = asset.data.joint_acc[0, self.leg_joint_ids]
        # The ordered list alternates left/right for each joint, so selecting
        # even/odd entries gives comparable six-joint left/right RMS values.
        left_leg_vel = leg_vel[0::2]
        right_leg_vel = leg_vel[1::2]
        left_leg_acc = leg_acc[0::2]
        right_leg_acc = leg_acc[1::2]
        command_xy = command.cmd_linvel_b[0, :2]
        reference_w = command.surr_lin_vel_target[0, :, :2]
        reference_yaw = yaw_quat(root_quat).expand(reference_w.shape[0], -1)
        reference_b = quat_rotate_inverse(
            reference_yaw,
            torch.cat((reference_w, torch.zeros_like(reference_w[:, :1])), dim=-1),
        )[:, :2].mean(dim=0)
        actual_com = quat_rotate_inverse(
            yaw_only, asset.data.root_com_lin_vel_w[0]
        )[:2]
        root_ang_vel_b = quat_rotate_inverse(
            root_quat, asset.data.root_link_ang_vel_w[0]
        )
        root_yaw_rate = asset.data.root_link_ang_vel_w[0, 2]
        command_heading = command.command_heading_w[0]
        heading_error = wrap_to_pi(asset.data.heading_w[0] - command_heading)

        left_contact = bool(contacts[0].item())
        right_contact = bool(contacts[1].item())
        return {
            "control_step": control_step,
            "sim_time_s": control_step * float(self.env.step_dt),
            "episode_id": episode_id,
            "episode_step": episode_step,
            "transition_done": "",
            "gait_phase": phase,
            "left_phase": foot_phases[0],
            "right_phase": foot_phases[1],
            "left_expected_stance": int(expected_stance[0]),
            "right_expected_stance": int(expected_stance[1]),
            "left_contact": int(left_contact),
            "right_contact": int(right_contact),
            "left_air_time_s": self._float(
                self.contact_sensor.data.current_air_time[0, self.contact_ids[0]]
            ),
            "right_air_time_s": self._float(
                self.contact_sensor.data.current_air_time[0, self.contact_ids[1]]
            ),
            "left_foot_x_yaw_m": self._float(foot_pos_b[0, 0]),
            "left_foot_y_yaw_m": self._float(foot_pos_b[0, 1]),
            "left_foot_z_m": self._float(foot_pos_b[0, 2]),
            "right_foot_x_yaw_m": self._float(foot_pos_b[1, 0]),
            "right_foot_y_yaw_m": self._float(foot_pos_b[1, 1]),
            "right_foot_z_m": self._float(foot_pos_b[1, 2]),
            "signed_lateral_gap_m": self._float(foot_pos_b[0, 1] - foot_pos_b[1, 1]),
            "foot_center_distance_m": self._float(
                (foot_pos_w[0] - foot_pos_w[1]).norm()
            ),
            "collision_mesh_sampled_surface_gap_est_m": mesh_gap_est,
            "left_mesh_vertex_inside_right": int(left_inside),
            "right_mesh_vertex_inside_left": int(right_inside),
            "left_touchdown": int(touchdowns[0]),
            "right_touchdown": int(touchdowns[1]),
            "left_touchdown_x_yaw_m": self._float(foot_pos_b[0, 0]) if touchdowns[0] else "",
            "left_touchdown_y_yaw_m": self._float(foot_pos_b[0, 1]) if touchdowns[0] else "",
            "right_touchdown_x_yaw_m": self._float(foot_pos_b[1, 0]) if touchdowns[1] else "",
            "right_touchdown_y_yaw_m": self._float(foot_pos_b[1, 1]) if touchdowns[1] else "",
            "left_step_forward_m": touchdown_data[0][0],
            "right_step_forward_m": touchdown_data[1][0],
            "left_touchdown_interval_s": touchdown_data[0][1],
            "right_touchdown_interval_s": touchdown_data[1][1],
            "left_knee_angle_rad": self._float(knees[0]),
            "right_knee_angle_rad": self._float(knees[1]),
            "left_knee_offset_rad": self._float(knees[0] - default_knees[0]),
            "right_knee_offset_rad": self._float(knees[1] - default_knees[1]),
            "left_hip_pitch_angle_rad": self._float(hips[0]),
            "right_hip_pitch_angle_rad": self._float(hips[1]),
            "left_hip_pitch_offset_rad": self._float(hips[0] - default_hips[0]),
            "right_hip_pitch_offset_rad": self._float(hips[1] - default_hips[1]),
            "left_hip_pitch_vel_radps": self._float(leg_vel[0]),
            "right_hip_pitch_vel_radps": self._float(leg_vel[1]),
            "left_hip_roll_vel_radps": self._float(leg_vel[2]),
            "right_hip_roll_vel_radps": self._float(leg_vel[3]),
            "left_hip_yaw_vel_radps": self._float(leg_vel[4]),
            "right_hip_yaw_vel_radps": self._float(leg_vel[5]),
            "left_knee_vel_radps": self._float(leg_vel[6]),
            "right_knee_vel_radps": self._float(leg_vel[7]),
            "left_ankle_pitch_vel_radps": self._float(leg_vel[8]),
            "right_ankle_pitch_vel_radps": self._float(leg_vel[9]),
            "left_ankle_roll_vel_radps": self._float(leg_vel[10]),
            "right_ankle_roll_vel_radps": self._float(leg_vel[11]),
            "left_hip_pitch_acc_radps2": self._float(leg_acc[0]),
            "right_hip_pitch_acc_radps2": self._float(leg_acc[1]),
            "left_hip_roll_acc_radps2": self._float(leg_acc[2]),
            "right_hip_roll_acc_radps2": self._float(leg_acc[3]),
            "left_hip_yaw_acc_radps2": self._float(leg_acc[4]),
            "right_hip_yaw_acc_radps2": self._float(leg_acc[5]),
            "left_knee_acc_radps2": self._float(leg_acc[6]),
            "right_knee_acc_radps2": self._float(leg_acc[7]),
            "left_ankle_pitch_acc_radps2": self._float(leg_acc[8]),
            "right_ankle_pitch_acc_radps2": self._float(leg_acc[9]),
            "left_ankle_roll_acc_radps2": self._float(leg_acc[10]),
            "right_ankle_roll_acc_radps2": self._float(leg_acc[11]),
            "left_leg_joint_vel_rms_radps": self._float(left_leg_vel.square().mean().sqrt()),
            "right_leg_joint_vel_rms_radps": self._float(right_leg_vel.square().mean().sqrt()),
            "left_leg_joint_acc_rms_radps2": self._float(left_leg_acc.square().mean().sqrt()),
            "right_leg_joint_acc_rms_radps2": self._float(right_leg_acc.square().mean().sqrt()),
            "left_foot_pitch_rel_torso_rad": self._float(foot_relative_rpy[0, 1]),
            "right_foot_pitch_rel_torso_rad": self._float(foot_relative_rpy[1, 1]),
            "left_foot_roll_rel_torso_rad": self._float(foot_relative_rpy[0, 0]),
            "right_foot_roll_rel_torso_rad": self._float(foot_relative_rpy[1, 0]),
            "foot_roll_difference_rad": self._float(
                foot_relative_rpy[0, 0] - foot_relative_rpy[1, 0]
            ),
            "foot_roll_mean_rad": self._float(foot_relative_rpy[:, 0].mean()),
            "left_foot_yaw_error_rad": self._float(foot_yaw_error[0]),
            "right_foot_yaw_error_rad": self._float(foot_yaw_error[1]),
            "left_foot_vertical_vel_mps": self._float(foot_lin_vel_w[0, 2]),
            "right_foot_vertical_vel_mps": self._float(foot_lin_vel_w[1, 2]),
            "root_com_y_m": self._float(root_com_pos[1]),
            "root_com_height_m": self._float(root_com_pos[2]),
            "root_roll_rad": self._float(root_rpy[0]),
            "root_pitch_rad": self._float(root_rpy[1]),
            "command_vx_yaw_mps": self._float(command_xy[0]),
            "command_vy_yaw_mps": self._float(command_xy[1]),
            "facet_reference_vx_yaw_mps": self._float(reference_b[0]),
            "facet_reference_vy_yaw_mps": self._float(reference_b[1]),
            "actual_com_vx_yaw_mps": self._float(actual_com[0]),
            "actual_com_vy_yaw_mps": self._float(actual_com[1]),
            "root_yaw_rad": self._float(asset.data.heading_w[0]),
            "command_heading_w_rad": self._float(command_heading),
            "heading_error_rad": self._float(heading_error),
            "root_roll_rate_radps": self._float(root_ang_vel_b[0]),
            "root_pitch_rate_radps": self._float(root_ang_vel_b[1]),
            "root_yaw_rate_radps": self._float(root_yaw_rate),
        }

    def write(self, row):
        self.writer.writerow(row)

    def close(self):
        self.file.close()


def transition_done(td):
    for key in (("next", "done"), "done"):
        try:
            value = td.get(key)
        except (AttributeError, KeyError):
            value = None
        if value is not None:
            return int(torch.as_tensor(value).reshape(-1)[0].item())
    return ""


def update_follow_camera(env, cfg):
    """Keep the Isaac camera centered on env 0's robot root during playback."""
    if not bool(cfg.get("camera_follow", False)):
        return
    if getattr(env, "backend", None) != "isaac":
        return

    base_env = getattr(env, "base_env", env)
    try:
        root_pos = base_env.scene.articulations["robot"].data.root_pos_w[0]
        offset = torch.as_tensor(cfg.camera_offset, device=root_pos.device, dtype=root_pos.dtype)
        lookat_offset = torch.as_tensor(
            cfg.camera_lookat_offset, device=root_pos.device, dtype=root_pos.dtype
        )
        eye = (root_pos + offset).detach().cpu().tolist()
        target = (root_pos + lookat_offset).detach().cpu().tolist()
        env.sim.set_camera_view(eye=eye, target=target)
    except (AttributeError, KeyError, RuntimeError):
        # Camera following is diagnostic/visualization only; it must not stop
        # policy playback if a backend wrapper lacks root pose access.
        return


class ComTrajectoryOverlay:
    """Render the FACET CoM preview and the robot's recent ground track."""

    def __init__(self, env, cfg):
        self.enabled = bool(cfg.get("show_com_trajectories", False))
        self.env = getattr(env, "base_env", env)
        self.update_interval = max(1, int(cfg.get("com_trajectory_update_interval", 4)))
        self.max_points = max(2, int(cfg.get("com_trajectory_max_points", 160)))
        self.step_count = 0
        self.episode_id = None
        self.actual_points = deque(maxlen=self.max_points)
        if not self.enabled:
            return
        if self.env.backend != "isaac":
            raise ValueError("show_com_trajectories requires backend=isaac")
        if self.env.num_envs != 1:
            raise ValueError("show_com_trajectories requires task.num_envs=1")

        self.command = self.env.command_manager
        if not hasattr(self.command, "surr_pos_target"):
            raise ValueError("The selected command manager has no FACET CoM reference")
        self.asset = self.env.scene.articulations["robot"]
        self.reference_marker = self.env.scene.create_sphere_marker(
            "/Visuals/Play/FacetComReferencePreview",
            color=(1.0, 0.0, 0.0),
            radius=0.018,
        )
        self.force_marker = self.env.scene.create_arrow_marker(
            "/Visuals/Play/AppliedDisturbanceForce",
            color=(1.0, 0.0, 0.0),
            # Runtime scales below control the arrow dimensions. Keep the
            # marker prototype at unit scale to avoid multiplying thickness
            # by 0.06 twice and making the arrow effectively invisible.
            scale=(1.0, 1.0, 1.0),
        )
        self.force_marker.set_visibility(False)
        self.actual_marker = self.env.scene.create_sphere_marker(
            "/Visuals/Play/ActualComTrail",
            color=(0.05, 0.85, 1.0),
            radius=0.022,
        )
        print(
            "CoM overlay enabled: red=future FACET reference samples, "
            "cyan=recent actual CoM trail; red arrow=applied disturbance "
            "at the robot root link."
        )

    @staticmethod
    def _x_axis_to_quaternion(direction: torch.Tensor) -> torch.Tensor:
        """Return a wxyz quaternion rotating local +X onto a world direction."""
        direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        x_axis = torch.zeros_like(direction)
        x_axis[..., 0] = 1.0
        cross = torch.cross(x_axis, direction, dim=-1)
        dot = (x_axis * direction).sum(dim=-1, keepdim=True)
        quat = torch.cat((1.0 + dot, cross), dim=-1)
        opposite = dot.squeeze(-1) < -0.9999
        if opposite.any():
            quat[opposite] = torch.tensor(
                [0.0, 0.0, 0.0, 1.0], device=quat.device, dtype=quat.dtype
            )
        return quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def update(self, step: int):
        if not self.enabled:
            return
        self.step_count += 1
        episode_id = int(self.env.episode_id[0].item())
        if episode_id != self.episode_id:
            self.episode_id = episode_id
            self.actual_points.clear()

        if self.step_count % self.update_interval != 0:
            return

        actual = self.asset.data.root_com_pos_w[0].detach().clone()
        self.actual_points.append(actual)
        reference = self.command.surr_pos_target[0].detach()
        preview = torch.cat((actual.unsqueeze(0), reference), dim=0)
        self.reference_marker.visualize(
            translations=preview
        )
        actual_trail = torch.stack(tuple(self.actual_points))
        self.actual_marker.visualize(
            translations=actual_trail
        )

        if hasattr(self.command, "current_external_force"):
            force = self.command.current_external_force[0].detach()
        else:
            force = self.command.force_impulse.get_force(None, None)[0].detach()
        force_norm = force.norm()
        if force_norm.item() > 1e-3:
            self.force_marker.set_visibility(True)
            force_direction = force.unsqueeze(0)
            force_quat = self._x_axis_to_quaternion(force_direction)
            # Keep moderate forces visible in the recorded camera view while
            # preserving their direction and relative magnitude.
            arrow_length = (force_norm / 50.0).clamp(0.25, 1.5)
            arrow_scales = torch.stack(
                (
                    arrow_length,
                    torch.full_like(arrow_length, 0.06),
                    torch.full_like(arrow_length, 0.06),
                ),
                dim=-1,
            ).unsqueeze(0)
            self.force_marker.visualize(
                translations=self.asset.data.root_link_pos_w[0:1].detach(),
                orientations=force_quat,
                scales=arrow_scales,
            )
        else:
            self.force_marker.set_visibility(False)


@VecNorm.freeze()
def export_policy(env, policy, export_dir):
    fake_input = env.observation_spec[0].rand().cpu()
    fake_input = fake_input.unsqueeze(0)

    deploy_policy = copy.deepcopy(policy.get_rollout_policy("deploy")).cpu()

    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    path = export_dir / f"policy-{time_str}.onnx"
    export_onnx(deploy_policy, fake_input, str(path))


@hydra.main(config_path=str(CONFIG_PATH), config_name="play", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    aa.init(cfg, auto_rank=True)
    
    from active_adaptation.helpers import make_env_policy
    checkpoint = parse_checkpoint(cfg.checkpoint_path)
    env, policy = make_env_policy(cfg, checkpoint)
    
    if cfg.export_policy:
        export_dir = FILE_PATH / "exports" / str(cfg.task.name)
        export_policy(env, policy, export_dir)

    stats_keys = [
        k for k in env.reward_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)
    rollout_policy = policy.get_rollout_policy("eval")
    
    env.base_env.eval()
    carry = env.reset()
    
    assert not env.base_env.training

    timer = Timer(env.step_dt)
    com_trajectory_overlay = ComTrajectoryOverlay(env, cfg)

    # Optional: refresh from URL/wandb in background so play loop never blocks on updates
    if checkpoint is not None and checkpoint.remote:
        print("Starting background checkpoint refresh")
        checkpoint.start_background_refresh(interval_sec=60)

    # Optional video recording (Isaac backend only). This remains safe under
    # KeyboardInterrupt because the recorder is a context manager that flushes
    # buffered frames on exit.
    record_enabled = bool(cfg.get("record_video", False))
    record_steps = int(cfg.get("record_steps", 0))
    video_dir = FILE_PATH / "videos"
    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    video_path = video_dir / f"{cfg.task.name}-{time_str}.mp4"
    gait_csv_enabled = bool(cfg.get("record_gait_csv", False))
    if gait_csv_enabled:
        gait_csv_path = cfg.get("gait_csv_path")
        if not gait_csv_path:
            gait_csv_path = Path("outputs_play") / f"{cfg.task.name}-{time_str}-gait.csv"
        gait_csv_recorder = GaitCsvRecorder(env, Path(gait_csv_path), cfg.task)
    else:
        gait_csv_recorder = None

    gait_csv_context = gait_csv_recorder or nullcontext()
    with env.get_recorder(video_path, enabled=record_enabled) as rec, \
        gait_csv_context, \
        torch.inference_mode(), set_exploration_type(ExplorationType.MODE):
        try:
            for i in itertools.count():
                gait_row = gait_csv_recorder.capture(i) if gait_csv_recorder else None
                carry = rollout_policy(carry)
                td, carry = env.step_and_maybe_reset(carry)
                episode_stats.add(td)
                com_trajectory_overlay.update(i)

                if gait_row is not None:
                    gait_row["transition_done"] = transition_done(td)
                    gait_csv_recorder.write(gait_row)

                update_follow_camera(env, cfg)

                if record_enabled:
                    rec.add_frame()

                if (i + 1) % 100 == 0:
                    target = record_steps if record_steps > 0 else "until interrupted"
                    print(
                        f"Recorded {i + 1}/{target} control steps", flush=True
                    )

                if record_steps > 0 and i + 1 >= record_steps and (
                    record_enabled or gait_csv_enabled
                ):
                    print(f"Finished recording {record_steps} control steps")
                    break

                if len(episode_stats) >= env.num_envs:
                    print("Step", i)
                    for k, v in sorted(episode_stats.pop().items(True, True)):
                        print(k, torch.mean(v).item())

                timer.sleep()
        except KeyboardInterrupt:
            print(f"Interrupted by user, video saved to: {video_path}" if record_enabled else "Interrupted by user.")
    
    env.close()


if __name__ == "__main__":
    main()
