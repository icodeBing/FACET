import torch
import einops

from active_adaptation.envs.mdp.rewards.base import Reward
from active_adaptation.utils.math import wrap_to_pi
from active_adaptation.utils.math import euler_from_quat
from .impedance_manip import ImpedanceCommandManager


class facet_external_force_norm(Reward[ImpedanceCommandManager]):
    """Diagnostic only: magnitude of the force currently applied at pelvis."""
    def __init__(self, env, weight: float = 1.0, enabled: bool = False):
        super().__init__(env, weight, enabled)

    def _compute(self) -> torch.Tensor:
        force = self.command_manager.force_impulse.get_force(None, None)
        return force.norm(dim=-1, keepdim=True)


class facet_external_force_active(Reward[ImpedanceCommandManager]):
    """Diagnostic only: 1 while the sampled pelvis impulse is active."""
    def __init__(self, env, weight: float = 1.0, enabled: bool = False):
        super().__init__(env, weight, enabled)

    def _compute(self) -> torch.Tensor:
        force = self.command_manager.force_impulse.get_force(None, None)
        return (force.norm(dim=-1, keepdim=True) > 1e-5).float()


class facet_external_force_component(Reward[ImpedanceCommandManager]):
    """Diagnostic only: signed world-frame pelvis force component."""
    def __init__(self, env, axis: int, weight: float = 1.0, enabled: bool = False):
        super().__init__(env, weight, enabled)
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2")
        self.axis = axis

    def _compute(self) -> torch.Tensor:
        force = self.command_manager.force_impulse.get_force(None, None)
        return force[:, self.axis:self.axis + 1]


class facet_pelvis_ref_pos_error(Reward[ImpedanceCommandManager]):
    """Diagnostic only: xy distance between the real CoM and FACET reference."""
    def __init__(self, env, weight: float = 1.0, enabled: bool = False):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset

    def _compute(self) -> torch.Tensor:
        target = self.command_manager.surr_pos_target[:, 0, :2]
        current = self.asset.data.root_com_pos_w[:, :2]
        return (target - current).norm(dim=-1, keepdim=True)


class facet_pelvis_ref_vel_error(Reward[ImpedanceCommandManager]):
    """Diagnostic only: xy CoM velocity tracking error to the FACET reference."""
    def __init__(self, env, weight: float = 1.0, enabled: bool = False):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset

    def _compute(self) -> torch.Tensor:
        target = self.command_manager.surr_lin_vel_target[:, 0, :2]
        current = self.asset.data.root_com_lin_vel_w[:, :2]
        return (target - current).norm(dim=-1, keepdim=True)


class facet_track_lin_vel(Reward[ImpedanceCommandManager]):
    """Nominal walking anchor: track the commanded CoM velocity without force.

    FACET's primary objective remains reference-model tracking.  This small
    auxiliary term is active only when no external impulse is present, so it
    prevents a stationary solution during nominal gait learning without
    making the policy reject the compliant response during a disturbance.
    """

    def __init__(self, env, weight: float, sigma: float = 0.25):
        super().__init__(env, weight)
        self.asset = self.command_manager.asset
        self.sigma = float(sigma)

    def _compute(self) -> torch.Tensor:
        command = self.command_manager.cmd_linvel_w[:, :2]
        current = self.asset.data.root_com_lin_vel_w[:, :2]
        error_l2 = (current - command).square().sum(dim=-1, keepdim=True)
        reward = torch.exp(-error_l2 / self.sigma)
        force_active = self.command_manager.force_impulse.get_force(None, None).norm(
            dim=-1, keepdim=True
        ) > 1e-5
        active = (~self.command_manager.is_standing_env) & (~force_active)
        return reward, active


class facet_command_linvel_x(Reward[ImpedanceCommandManager]):
    """Diagnostic: commanded forward velocity in the world frame (m/s)."""

    def __init__(self, env, weight: float = 1.0, enabled: bool = False):
        super().__init__(env, weight, enabled)

    def _compute(self):
        return self.command_manager.cmd_linvel_w[:, 0:1]


class facet_reference_linvel_x(Reward[ImpedanceCommandManager]):
    """Diagnostic: mean forward velocity of the smoothed FACET targets."""

    def __init__(self, env, weight: float = 1.0, enabled: bool = False):
        super().__init__(env, weight, enabled)

    def _compute(self):
        return self.command_manager.surr_lin_vel_target[:, :, 0].mean(
            dim=1, keepdim=True
        )


class facet_actual_com_linvel_x(Reward[ImpedanceCommandManager]):
    """Diagnostic: actual CoM forward velocity in the world frame (m/s)."""

    def __init__(self, env, weight: float = 1.0, enabled: bool = False):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset

    def _compute(self):
        return self.asset.data.root_com_lin_vel_w[:, 0:1]


class facet_command_reference_vel_error(Reward[ImpedanceCommandManager]):
    """Diagnostic: commanded velocity versus the smoothed reference target."""

    def __init__(self, env, weight: float = 1.0, enabled: bool = False):
        super().__init__(env, weight, enabled)

    def _compute(self):
        command = self.command_manager.cmd_linvel_w[:, :2].unsqueeze(1)
        target = self.command_manager.surr_lin_vel_target[:, :, :2]
        return (target - command).norm(dim=-1).mean(dim=1, keepdim=True)


class facet_command_actual_vel_error(Reward[ImpedanceCommandManager]):
    """Diagnostic: commanded velocity versus actual CoM velocity."""

    def __init__(self, env, weight: float = 1.0, enabled: bool = False):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset

    def _compute(self):
        command = self.command_manager.cmd_linvel_w[:, :2]
        current = self.asset.data.root_com_lin_vel_w[:, :2]
        return (current - command).norm(dim=-1, keepdim=True)


class facet_root_link_com_vel_delta(Reward[ImpedanceCommandManager]):
    """Diagnostic: root-link versus CoM velocity discrepancy."""

    def __init__(self, env, weight: float = 1.0, enabled: bool = False):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset

    def _compute(self):
        delta = (
            self.asset.data.root_link_lin_vel_w[:, :2]
            - self.asset.data.root_com_lin_vel_w[:, :2]
        )
        return delta.norm(dim=-1, keepdim=True)


class facet_root_link_com_pos_delta(Reward[ImpedanceCommandManager]):
    """Diagnostic: root-link versus CoM position discrepancy in xy (m)."""

    def __init__(self, env, weight: float = 1.0, enabled: bool = False):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset

    def _compute(self):
        delta = (
            self.asset.data.root_link_pos_w[:, :2]
            - self.asset.data.root_com_pos_w[:, :2]
        )
        return delta.norm(dim=-1, keepdim=True)


class facet_pelvis_pitch_roll_abs(Reward[ImpedanceCommandManager]):
    """Diagnostic only: absolute pelvis roll/pitch angle in radians."""
    def __init__(self, env, weight: float = 1.0, enabled: bool = False):
        super().__init__(env, weight, enabled)
        self.asset = self.command_manager.asset

    def _compute(self) -> torch.Tensor:
        rpy = euler_from_quat(self.asset.data.root_link_quat_w)
        return rpy[:, :2].abs().norm(dim=-1, keepdim=True)


class facet_leg_joint_pos_dev(Reward):
    """Diagnostic only: RMS leg joint deviation from the default pose."""
    def __init__(self, env, joint_names, weight: float = 1.0, enabled: bool = False):
        super().__init__(env, weight, enabled)
        self.asset = env.command_manager.asset
        self.joint_ids = self.asset.find_joints(joint_names)[0]

    def _compute(self) -> torch.Tensor:
        error = self.asset.data.joint_pos[:, self.joint_ids] - self.asset.data.default_joint_pos[:, self.joint_ids]
        return error.square().mean(dim=-1, keepdim=True).sqrt()


class facet_leg_joint_vel_rms(Reward):
    """Diagnostic only: RMS velocity of leg joints."""
    def __init__(self, env, joint_names, weight: float = 1.0, enabled: bool = False):
        super().__init__(env, weight, enabled)
        self.asset = env.command_manager.asset
        self.joint_ids = self.asset.find_joints(joint_names)[0]

    def _compute(self) -> torch.Tensor:
        velocity = self.asset.data.joint_vel[:, self.joint_ids]
        return velocity.square().mean(dim=-1, keepdim=True).sqrt()


class _FacetFootDiagnostic(Reward):
    """Base for scalar diagnostics attached to one ordered foot."""

    def __init__(self, env, body_names, foot_index: int = 0, weight: float = 1.0, enabled: bool = False):
        super().__init__(env, weight, enabled)
        self.contact_sensor = env.scene.sensors["contact_forces"]
        body_ids, _ = self.contact_sensor.find_bodies(body_names, preserve_order=True)
        if len(body_ids) != 2 or foot_index not in (0, 1):
            raise ValueError("foot diagnostics require two ordered foot bodies")
        self.body_id = body_ids[foot_index]
        self.foot_index = foot_index

    def _contact(self):
        return self.contact_sensor.data.current_contact_time[:, self.body_id:self.body_id + 1] > 0.0


class facet_foot_contact(_FacetFootDiagnostic):
    """Instantaneous left/right foot contact state (0 or 1)."""

    def _compute(self):
        return self._contact().float()


class facet_foot_air_time(_FacetFootDiagnostic):
    """Current air time; it resets to zero at touchdown."""

    def _compute(self):
        return self.contact_sensor.data.current_air_time[:, self.body_id:self.body_id + 1]


class facet_foot_touchdown_interval(_FacetFootDiagnostic):
    """Time between consecutive touchdowns for the selected foot."""

    def __init__(self, env, body_names, foot_index: int = 0, weight: float = 1.0, enabled: bool = False):
        super().__init__(env, body_names, foot_index, weight, enabled)
        self.prev_contact = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        self.last_touchdown = torch.full((self.num_envs, 1), -1.0, device=self.device)
        self.interval = torch.zeros(self.num_envs, 1, device=self.device)

    def reset(self, env_ids):
        self.prev_contact[env_ids] = False
        self.last_touchdown[env_ids] = -1.0
        self.interval[env_ids] = 0.0

    def _compute(self):
        contact = self._contact()
        touchdown = contact & ~self.prev_contact
        now = self.env.episode_length_buf.to(torch.float32).unsqueeze(1) * self.env.step_dt
        valid = touchdown & (self.last_touchdown >= 0.0)
        self.interval = torch.where(valid, now - self.last_touchdown, self.interval)
        self.last_touchdown = torch.where(touchdown, now, self.last_touchdown)
        self.prev_contact = contact
        return self.interval


class facet_foot_step_length(_FacetFootDiagnostic):
    """Horizontal distance travelled by a foot between touchdowns."""

    def __init__(self, env, body_names, foot_index: int = 0, weight: float = 1.0, enabled: bool = False):
        super().__init__(env, body_names, foot_index, weight, enabled)
        self.asset = env.command_manager.asset
        self.prev_contact = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        self.last_touchdown_pos = torch.zeros(self.num_envs, 2, device=self.device)
        self.step_length = torch.zeros(self.num_envs, 1, device=self.device)

    def reset(self, env_ids):
        self.prev_contact[env_ids] = False
        self.last_touchdown_pos[env_ids] = self.asset.data.body_pos_w[env_ids, self.body_id, :2]
        self.step_length[env_ids] = 0.0

    def _compute(self):
        contact = self._contact()
        touchdown = contact & ~self.prev_contact
        pos = self.asset.data.body_pos_w[:, self.body_id, :2]
        distance = (pos - self.last_touchdown_pos).norm(dim=-1, keepdim=True)
        self.step_length = torch.where(touchdown, distance, self.step_length)
        self.last_touchdown_pos = torch.where(touchdown, pos, self.last_touchdown_pos)
        self.prev_contact = contact
        return self.step_length


class facet_foot_knee_peak_flexion(_FacetFootDiagnostic):
    """Peak knee flexion offset observed during the current swing."""

    def __init__(self, env, body_names, knee_names, foot_index: int = 0, weight: float = 1.0, enabled: bool = False):
        super().__init__(env, body_names, foot_index, weight, enabled)
        self.asset = env.command_manager.asset
        knee_ids, _ = self.asset.find_joints(knee_names, preserve_order=True)
        if len(knee_ids) != 2:
            raise ValueError("knee diagnostics require two ordered knee joints")
        self.knee_id = knee_ids[foot_index]
        self.default_knee = self.asset.data.default_joint_pos[:, self.knee_id:self.knee_id + 1]
        self.prev_contact = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        self.peak = torch.zeros(self.num_envs, 1, device=self.device)

    def reset(self, env_ids):
        self.prev_contact[env_ids] = False
        self.peak[env_ids] = 0.0

    def _compute(self):
        contact = self._contact()
        knee_flexion = (self.asset.data.joint_pos[:, self.knee_id:self.knee_id + 1] - self.default_knee).clamp_min(0.0)
        self.peak = torch.where(contact, self.peak, torch.maximum(self.peak, knee_flexion))
        touchdown = contact & ~self.prev_contact
        self.peak = torch.where(touchdown, knee_flexion, self.peak)
        self.prev_contact = contact
        return self.peak


class facet_root_y_position(Reward):
    def __init__(self, env, weight: float = 1.0, enabled: bool = False):
        super().__init__(env, weight, enabled)
        self.asset = env.command_manager.asset

    def _compute(self):
        return self.asset.data.root_com_pos_w[:, 1:2]


class facet_root_yaw(Reward):
    def __init__(self, env, weight: float = 1.0, enabled: bool = False):
        super().__init__(env, weight, enabled)
        self.asset = env.command_manager.asset

    def _compute(self):
        return self.asset.data.heading_w[:, None]


class facet_root_yaw_rate(Reward):
    def __init__(self, env, weight: float = 1.0, enabled: bool = False):
        super().__init__(env, weight, enabled)
        self.asset = env.command_manager.asset

    def _compute(self):
        return self.asset.data.root_link_ang_vel_w[:, 2:3]


class impedance_pos(Reward[ImpedanceCommandManager]):
    def __init__(self, env, weight: float):
        super().__init__(env, weight)
        self.asset = self.command_manager.asset

    def _compute(self) -> torch.Tensor:
        target_pos_xy = self.command_manager.surr_pos_target[:, :, :2]
        current_pos_xy = self.asset.data.root_com_pos_w[:, :2].unsqueeze(1)
        diff = target_pos_xy - current_pos_xy
        error_l2 = diff.square().sum(dim=-1, keepdim=True)
        r = (- error_l2 / 0.25).exp().mean(1)
        return r.reshape(self.num_envs, 1)


class impedance_eef_pos(Reward[ImpedanceCommandManager]):
    def __init__(self, env, weight: float):
        super().__init__(env, weight)
        self.asset = self.command_manager.asset
        self.eef_body_id = self.command_manager.eef_body_id

    def _compute(self) -> torch.Tensor:
        target_pos_xy = self.command_manager.surr_eef_pos_target
        current_pos_xy = self.asset.data.body_pos_w[:, self.eef_body_id].unsqueeze(1)
        diff = target_pos_xy - current_pos_xy
        error_l2 = diff.square().sum(dim=-1, keepdim=True)
        r = (- error_l2 / 0.1).exp().mean(1)
        return r.reshape(self.num_envs, 1)


class impedance_vel(Reward[ImpedanceCommandManager]):
    def __init__(self, env, weight: float):
        super().__init__(env, weight)
        self.asset = self.command_manager.asset

    def _compute(self) -> torch.Tensor:
        target_vel_xy = self.command_manager.surr_lin_vel_target[:, :, :2]
        diff = target_vel_xy - self.asset.data.root_com_lin_vel_w[:, :2].unsqueeze(1)
        error_l2 = diff.square().sum(dim=-1, keepdim=True)
        r = ((- error_l2 / 0.25).exp() - 0.25 * error_l2).mean(1)
        return r.reshape(self.num_envs, 1)


class impedance_eef_vel(Reward[ImpedanceCommandManager]):
    def __init__(self, env, weight: float):
        super().__init__(env, weight)
        self.asset = self.command_manager.asset
        self.eef_body_id = self.command_manager.eef_body_id

    def _compute(self) -> torch.Tensor:
        target_vel_xy = self.command_manager.surr_eef_lin_vel_target
        current_vel_xy = self.asset.data.body_lin_vel_w[:, self.eef_body_id].unsqueeze(1)
        diff = target_vel_xy - current_vel_xy
        error_l2 = diff.square().sum(dim=-1, keepdim=True)
        r = ((- error_l2 / 0.25).exp()).mean(1)
        return r.reshape(self.num_envs, 1)


class impedance_acc(Reward):
    def __init__(self, env, weight: float):
        super().__init__(env, weight)
        self.command_manager: ImpedanceCommandManager = self.env.command_manager
        self.asset = self.command_manager.asset

    def _compute(self) -> torch.Tensor:
        lin_acc_w = self.asset.data.body_acc_w[:, 0, :2]
        ref_lin_acc_w = self.command_manager.pos_base.acc_w[:, 0, :2]
        error_l2 = (ref_lin_acc_w - lin_acc_w).square().sum(1, True)
        return torch.exp(- error_l2 / 2.0)


class impedance_yaw_pos(Reward[ImpedanceCommandManager]):
    def __init__(self, env, weight: float):
        super().__init__(env, weight)
        self.asset = self.command_manager.asset
        
    def _compute(self) -> torch.Tensor:
        target_yaw = self.command_manager.surr_yaw_target 
        diff = target_yaw - self.asset.data.heading_w.reshape(-1, 1, 1)
        diff = wrap_to_pi(diff)
        error_l2 = diff.square()
        r = torch.exp(-error_l2 / 0.25).mean(1)
        return r


class impedance_yaw_vel(Reward[ImpedanceCommandManager]):
    def __init__(self, env, weight: float):
        super().__init__(env, weight)
        self.asset = self.command_manager.asset

    def _compute(self) -> torch.Tensor:
        target_yaw_vel = self.command_manager.surr_yaw_vel_target
        current_yaw_vel = self.asset.data.root_link_ang_vel_w[:, 2:3].unsqueeze(1)
        diff = target_yaw_vel - current_yaw_vel
        error_l2 = diff.square()
        r = ((- error_l2 / 0.25).exp() - 0.25 * error_l2).mean(1)
        return r.reshape(self.num_envs, 1)


class impedance_pos_error(Reward[ImpedanceCommandManager]):
    def __init__(self, env, weight: float):
        super().__init__(env, weight)
        self.asset = self.command_manager.asset

    def _compute(self) -> torch.Tensor:
        target = self.command_manager.surr_pos_target[:, :, :2]
        current = self.asset.data.root_link_pos_w[:, :2].unsqueeze(1)
        diff = target - current
        error_l2 = diff[:, :, :2].square().sum(dim=-1, keepdim=True)
        return error_l2.mean(1)


class impedance_vel_error(Reward[ImpedanceCommandManager]):
    def __init__(self, env, weight: float):
        super().__init__(env, weight)
        self.asset = self.command_manager.asset

    def _compute(self) -> torch.Tensor:
        target = self.command_manager.surr_lin_vel_target[:, :, :2]
        current = self.asset.data.root_link_lin_vel_w[:, :2].unsqueeze(1)
        diff = target - current
        error_l2 = diff[:, :, :2].square().sum(dim=-1, keepdim=True)
        return error_l2.mean(1)


# class impedance_acc_error(Reward[Impedance]):
#     def __init__(self, env, weight: float):
#         super().__init__(env, weight)
#         self.impedance: Impedance = self.env.command_manager

#     def compute(self) -> torch.Tensor:
#         diff = self.impedance.ref_lin_acc_w[:, 0] - self.impedance.asset.data.body_acc_w[:, 0, :3]
#         error_l2 = diff[:, :2].square().sum(dim=-1, keepdim=True)
#         return error_l2
