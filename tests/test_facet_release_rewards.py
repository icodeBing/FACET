"""Numerical regressions for drive tracking and dynamic release balance.

Run: python -m unittest discover -s tests -p test_facet_release_rewards.py -v
The real reward classes are loaded from source without starting Isaac Sim.
"""
import abc
import ast
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Generic, TypeVar
import unittest

import torch
import yaml

from test_facet_force_contract import MATH, load_function


ROOT = Path(__file__).resolve().parents[1]


def load_classes(path, names, scope):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    module = ast.Module(body=[ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    ), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), path, "exec"), scope)


SCOPE = {"torch": torch, "math": math, "abc": abc, "Generic": Generic, "CT": TypeVar("CT"),
         "RegistryMixin": type("RegistryMixin", (), {}),
         "ImpedanceCommandManager": type("ImpedanceCommandManager", (), {}), **MATH}
load_classes("active_adaptation/envs/mdp/base.py", {"MDPComponent"}, SCOPE)
load_classes("active_adaptation/envs/mdp/rewards/base.py", {"Reward"}, SCOPE)
SCOPE["_actual_com_linvel_h"] = load_function(
    "projects/facet/facet_commands/rewards.py", "_actual_com_linvel_h", MATH
)
load_classes("projects/facet/facet_commands/rewards.py", {
    "root_lateral_velocity_l2", "facet_force_velocity_tracking",
    "facet_foot_lateral_corridor", "facet_stand_mean_velocity",
}, SCOPE)
load_classes("active_adaptation/envs/mdp/rewards/gait.py", {"feet_lateral_separation"}, SCOPE)


class Force:
    def __init__(self, n):
        self.value = torch.zeros(n, 3)

    def get_force(self, *args):
        return self.value


class Manager(SimpleNamespace):
    current_external_force = property(load_function(
        "projects/facet/facet_commands/impedance_manip.py", "current_external_force",
        owner="ImpedanceCommandManager",
    ))
    disturbance_force_active = property(load_function(
        "projects/facet/facet_commands/impedance_manip.py", "disturbance_force_active",
        owner="ImpedanceCommandManager",
    ))
    impulse_force_active = property(load_function(
        "projects/facet/facet_commands/impedance_manip.py", "impulse_force_active",
        owner="ImpedanceCommandManager",
    ))


FOOT_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
KNEE_NAMES = ["left_knee_link", "right_knee_link"]


def make_env(n=1):
    body_names = FOOT_NAMES + KNEE_NAMES
    data = SimpleNamespace(
        root_com_lin_vel_w=torch.zeros(n, 3), root_com_lin_vel_b=torch.zeros(n, 3),
        root_link_pos_w=torch.zeros(n, 3),
        root_link_quat_w=MATH["quat_from_yaw"](torch.zeros(n)),
        body_link_pos_w=torch.zeros(n, 4, 3),
    )
    data.body_link_pos_w[:, :, 1] = torch.tensor([.10, -.10, .08, -.08])
    asset = SimpleNamespace(data=data, find_bodies=lambda names, **kwargs: (
        [body_names.index(name) for name in names], names
    ))
    command = Manager(
        asset=asset, command_heading_w=torch.zeros(n),
        cmd_linvel_h=torch.zeros(n, 3), cmd_linvel_b=torch.zeros(n, 3),
        is_standing_env=torch.zeros(n, 1, dtype=torch.bool),
        force_impulse=Force(n), force_drive=Force(n),
    )
    return SimpleNamespace(num_envs=n, device="cpu", step_dt=.02,
                           command_manager=command, extra={},
                           scene=SimpleNamespace(articulations={"robot": asset}))


class DriveTrackingTests(unittest.TestCase):
    def test_drive_is_not_an_independent_disturbance(self):
        env = make_env(2)
        c = env.command_manager
        c.force_drive.value[:, 0] = torch.tensor([8., -8.])
        self.assertTrue(c.disturbance_force_active.all())  # legacy API unchanged
        self.assertFalse(c.impulse_force_active.any())
        # Even cancellation of the total force must not hide an impulse.
        c.force_impulse.value[:, 0] = -c.force_drive.value[:, 0]
        self.assertFalse(c.disturbance_force_active.any())
        self.assertTrue(c.impulse_force_active.all())

    def test_lateral_gate_allows_drive_but_excludes_impulse_and_standing(self):
        env = make_env(3)
        c = env.command_manager
        c.force_drive.value[:, 0] = 8.
        c.force_impulse.value[1, 1] = 2.
        c.is_standing_env[2] = True
        reward = SCOPE["root_lateral_velocity_l2"](
            env, 1., allow_force_drive=True, reference_frame="task", velocity_sigma=.1
        )
        c.asset.data.root_com_lin_vel_w[:, 1] = .1
        values, mask = reward._compute()
        torch.testing.assert_close(values, -torch.ones(3, 1))
        self.assertEqual(mask[:, 0].tolist(), [True, False, False])
        self.assertFalse(SCOPE["root_lateral_velocity_l2"](env, 1.)._compute()[1].any())

    def test_signed_tracking_and_task_frame_are_independent_of_robot_heading(self):
        env = make_env(2)
        c = env.command_manager
        c.force_drive.value[:, 0] = torch.tensor([8., -8.])
        c.command_heading_w[:] = math.pi / 2
        c.cmd_linvel_h[:, 0] = torch.tensor([.25, -.25])
        c.asset.data.root_com_lin_vel_w[:, 1] = c.cmd_linvel_h[:, 0]
        c.asset.data.root_link_quat_w[:] = MATH["quat_from_yaw"](torch.tensor([0., math.pi]))
        reward = SCOPE["facet_force_velocity_tracking"](env, 4.)
        good, active = reward._compute()
        torch.testing.assert_close(good, torch.ones(2, 1))
        self.assertTrue(active.all())
        c.asset.data.root_com_lin_vel_w.zero_()
        stopped, _ = reward._compute()
        self.assertTrue((stopped < 0).all())
        c.asset.data.root_com_lin_vel_w[:, 1] = -c.cmd_linvel_h[:, 0]
        wrong, _ = reward._compute()
        self.assertTrue((wrong < stopped).all())
        c.force_impulse.value[0, 0] = 1.
        c.is_standing_env[1] = True
        values, count = reward.compute()
        torch.testing.assert_close(values, torch.zeros(2, 1))
        self.assertEqual(count.item(), 0)


class ReleaseMeanVelocityTests(unittest.TestCase):
    def make_case(self, n=1, **kwargs):
        env = make_env(n)
        env.command_manager.is_standing_env[:] = True
        return env, SCOPE["facet_stand_mean_velocity"](env, 2., **kwargs)

    def test_full_cycle_balance_motion_is_allowed_but_drift_is_penalized(self):
        env, reward = self.make_case(2)
        for i in range(80):
            velocity = env.command_manager.asset.data.root_com_lin_vel_w
            velocity[:, 0] = .18 * math.sin(2 * math.pi * i / 40)
            velocity[:, 1] = .07 * math.cos(2 * math.pi * i / 40)
            velocity[1, 0] += .05
            reward.update()
        values, active = reward._compute()
        self.assertTrue(active.all())
        self.assertAlmostEqual(values[0].item(), 0., places=5)
        self.assertLess(values[1].item(), -.8)
        self.assertAlmostEqual(reward.sample_count[0].item(), 40)
        # Repeated reads must not advance the history.
        torch.testing.assert_close(values, reward._compute()[0])

    def test_warmup_and_force_to_release_history_are_independent(self):
        env, reward = self.make_case()
        c = env.command_manager
        c.is_standing_env[:] = False
        c.asset.data.root_com_lin_vel_w[:, 0] = .25
        for _ in range(50):
            reward.update()
        c.is_standing_env[:] = True
        c.asset.data.root_com_lin_vel_w.zero_()
        for _ in range(19):
            reward.update()
        self.assertFalse(reward._compute()[1].any())
        reward.update()
        values, active = reward._compute()
        self.assertTrue(active.all())
        torch.testing.assert_close(values, torch.zeros(1, 1))

    def test_reset_and_restart_clear_only_the_relevant_environment(self):
        env, reward = self.make_case(2)
        c = env.command_manager
        c.asset.data.root_com_lin_vel_w[:, 0] = .1
        for _ in range(60):
            reward.update()
        reward.reset(torch.tensor([0]))
        self.assertEqual(reward.sample_count[:, 0].tolist(), [0., 40.])
        self.assertAlmostEqual(reward.velocity_sum_w[0].norm().item(), 0.)
        c.is_standing_env[1] = False
        reward.update()
        self.assertEqual(reward.sample_count[:, 0].tolist(), [1., 0.])
        c.is_standing_env[1] = True
        c.asset.data.root_com_lin_vel_w[1] = 0.
        reward.update()
        torch.testing.assert_close(reward.velocity_sum_w[1], torch.zeros(2))


class LegGeometryTests(unittest.TestCase):
    def test_crossing_is_penalized_during_release_only_when_opted_in(self):
        env = make_env(2)
        env.command_manager.is_standing_env[:] = True
        env.command_manager.asset.data.body_link_pos_w[1, :2, 1] = torch.tensor([-.04, -.02])
        kwargs = dict(body_names=FOOT_NAMES, min_separation=.12,
                      knee_body_names=KNEE_NAMES, min_knee_separation=.10)
        old = SCOPE["feet_lateral_separation"](env, 2.25, **kwargs)
        new = SCOPE["feet_lateral_separation"](env, 2.25, include_standing=True, **kwargs)
        self.assertFalse(old._compute()[1].any())
        values, mask = new._compute()
        self.assertTrue(mask.all())
        self.assertEqual(values[0].item(), 0.)
        self.assertLess(values[1].item(), values[0].item())

    def test_corridor_detects_single_leg_detour_and_shared_side_bias(self):
        env = make_env(3)
        feet = env.command_manager.asset.data.body_link_pos_w
        feet[1, :2, 1] = torch.tensor([-.04, -.30])
        feet[2, :2, 1] = torch.tensor([-.08, -.28])  # width is still normal
        reward = SCOPE["facet_foot_lateral_corridor"](env, 1., FOOT_NAMES)
        values = reward._compute()
        self.assertEqual(values[0].item(), 0.)
        self.assertTrue((values[1:] < 0).all())
        # Fore/aft stepping is not constrained by the lateral corridor.
        feet[:, :, 0] = torch.tensor([.3, -.3, .1, -.1])
        torch.testing.assert_close(reward._compute(), values)
        # Global yaw/translation cannot change relative foot geometry.
        q = MATH["quat_from_yaw"](torch.full((3,), 1.2))
        inverse = MATH["quat_from_yaw"](torch.full((3,), -1.2))
        feet[:] = MATH["quat_rotate_inverse"](inverse.unsqueeze(1), feet)
        feet.add_(torch.tensor([3., -2., 1.]))
        env.command_manager.asset.data.root_link_pos_w[:] = torch.tensor([3., -2., 1.])
        env.command_manager.asset.data.root_link_quat_w[:] = q
        torch.testing.assert_close(reward._compute(), values, atol=1e-5, rtol=1e-5)

    def test_corridor_is_left_right_symmetric_and_allows_weight_transfer(self):
        env = make_env(2)
        feet = env.command_manager.asset.data.body_link_pos_w
        feet[0, :2, 1] = torch.tensor([-.015, -.18])
        feet[1, :2, 1] = torch.tensor([.18, .015])
        reward = SCOPE["facet_foot_lateral_corridor"](env, 1., FOOT_NAMES)
        torch.testing.assert_close(reward._compute(), torch.zeros(2, 1))
        feet[0, :2, 1] = torch.tensor([-.04, -.30])
        feet[1, :2, 1] = torch.tensor([.30, .04])
        values = reward._compute()
        torch.testing.assert_close(values[0], values[1])


class StageConfigurationTests(unittest.TestCase):
    def test_stage3_uses_dynamic_balance_without_changing_policy_interface(self):
        path = ROOT / "projects/facet/task/facet/G1LocoFacetForceBackwardStand.yaml"
        cfg = yaml.safe_load(path.read_text())
        rewards = cfg["reward"]["loco"]
        self.assertTrue(rewards["feet_lateral_separation"]["include_standing"])
        self.assertTrue(rewards["root_lateral_velocity_l2"]["allow_force_drive"])
        self.assertEqual(rewards["facet_backward_com_velocity_tracking"]["weight"], 0.)
        self.assertGreater(rewards["facet_force_velocity_tracking"]["weight"], 0.)
        for name in ["facet_stand_joint_posture", "facet_stand_foot_contact", "facet_stand_foot_velocity"]:
            self.assertEqual(rewards[name]["weight"], 0.)
        self.assertGreaterEqual(cfg["command"]["force_drive_interval_range"][1], 14.)
        self.assertNotIn("input", cfg)
        self.assertNotIn("observation", cfg)


if __name__ == "__main__":
    unittest.main()
