import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors import ContactSensor

from .base import Reward
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse, yaw_quat


class max_feet_height(Reward):
    def __init__(self, env, weight: float, body_names: str, target_height: float):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.body_ids = self.asset.find_bodies(body_names)[0]
        self.target_height = target_height

        self.command_manager = self.env.command_manager

    def _compute(self) -> torch.Tensor:
        feet_height = self.asset.data.body_link_pos_w[:, self.body_ids, 2]
        feet_velocity_xy = self.asset.data.body_lin_vel_w[:, self.body_ids, :2]
        speed_gate = torch.tanh(2.0 * feet_velocity_xy.norm(dim=-1))
        height_error = (feet_height - self.target_height).square()
        gravity_gate = (
            (-self.asset.data.projected_gravity_b[:, 2]).clamp(0.0, 0.7) / 0.7
        )
        moving = self.command_manager.command_speed.squeeze(-1) > 0.1
        cost = (height_error * speed_gate).sum(dim=1, keepdim=True) / 0.05
        reward = torch.exp(-cost) * gravity_gate.unsqueeze(1)
        return reward, moving.unsqueeze(1)


class feet_lateral_separation(Reward):
    """Penalize excessively wide lateral stance without forcing feet together."""

    def __init__(
        self,
        env,
        weight: float,
        body_names: list[str],
        max_separation: float = 0.28,
        scale: float = 0.08,
    ):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(
            body_names, preserve_order=True
        )
        if len(self.body_ids) != 2:
            raise ValueError("feet_lateral_separation requires two ordered foot bodies")
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.max_separation = float(max_separation)
        self.scale = float(scale)
        if self.max_separation <= 0.0 or self.scale <= 0.0:
            raise ValueError("max_separation and scale must be positive")

    def _compute(self) -> torch.Tensor:
        root_pos = self.asset.data.root_link_pos_w
        root_quat = self.asset.data.root_link_quat_w
        feet_pos_b = quat_rotate_inverse(
            root_quat.unsqueeze(1),
            self.asset.data.body_link_pos_w[:, self.body_ids] - root_pos.unsqueeze(1),
        )
        separation = (feet_pos_b[:, 0, 1] - feet_pos_b[:, 1, 1]).abs()
        excess = (separation - self.max_separation).clamp_min(0.0) / self.scale
        penalty = -excess.square().unsqueeze(1)
        active = ~self.command_manager.is_standing_env
        return penalty, active.reshape(self.num_envs, 1)


class feet_sliding(Reward):
    supported_backends = ("isaac",)
    def __init__(self, env, body_names: str, weight: float):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: ContactSensor = self.env.scene.sensors["contact_forces"]
        self.body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids = torch.tensor(self.body_ids, device=self.device)
        self.body_contact_ids = self.contact_sensor.find_bodies(body_names)[0]
        self.body_contact_ids = torch.tensor(self.body_contact_ids, device=self.device)

    def _compute(self) -> torch.Tensor:
        in_contact = self.contact_sensor.data.current_contact_time[:, self.body_contact_ids] > self.env.physics_dt 
        feet_speed = self.asset.data.body_lin_vel_w[:, self.body_ids].norm(dim=-1)
        sliding = (in_contact * feet_speed).sum(dim=1)
        return - sliding.reshape(self.num_envs, 1)


class quadruped_trot(Reward):
    """
    Reward either (FL-RR) or (FR-RL) are in contact but not both.
    """
    def __init__(self, env, weight: float, body_names: str):
        super().__init__(env, weight)
        self.asset: Articulation = self.env.scene.articulations["robot"]
        self.contact_sensor: ContactSensor = self.env.scene.sensors["contact_forces"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)

        self.body_contact_ids = self.contact_sensor.find_bodies(body_names)[0]
        self.body_contact_ids = torch.tensor(self.body_contact_ids, device=self.device)
    
    def _compute(self) -> torch.Tensor:
        in_contact = self.contact_sensor.data.current_contact_time[:, self.body_contact_ids] > 0.005
        FL_RR = in_contact[:, [0, 3]].all(dim=1)
        FR_RL = in_contact[:, [1, 2]].all(dim=1)
        rew = torch.logical_xor(FL_RR, FR_RL)
        active = ~self.command_manager.is_standing_env
        return rew.reshape(self.num_envs, 1), active.reshape(self.num_envs, 1)
