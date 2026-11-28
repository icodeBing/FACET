"""Numerical regressions for drive tracking and dynamic release balance.

Run: python -m unittest discover -s tests -p test_facet_release_rewards.py -v
The real reward classes are loaded from source without starting Isaac Sim.
"""
import abc
import ast
import importlib.util
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
    "facet_knee_extension_margin", "facet_stand_support_center", "facet_stand_leg_balance",
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
KNEE_JOINTS = ["left_knee_joint", "right_knee_joint"]
HIP_JOINTS = ["left_hip_pitch_joint", "right_hip_pitch_joint"]


def make_env(n=1):
    body_names = FOOT_NAMES + KNEE_NAMES
    data = SimpleNamespace(
        root_com_lin_vel_w=torch.zeros(n, 3), root_com_lin_vel_b=torch.zeros(n, 3),
        root_link_pos_w=torch.zeros(n, 3),
        root_com_pos_w=torch.zeros(n, 3),
        root_link_quat_w=MATH["quat_from_yaw"](torch.zeros(n)),
        body_link_pos_w=torch.zeros(n, 4, 3),
        joint_pos=torch.tensor([.3, .3, -.1, -.1]).expand(n, -1).clone(),
        default_joint_pos=torch.tensor([.3, .3, -.1, -.1]).expand(n, -1).clone(),
    )
    data.body_link_pos_w[:, :, 1] = torch.tensor([.10, -.10, .08, -.08])
    asset = SimpleNamespace(
        data=data,
        find_bodies=lambda names, **kwargs: ([body_names.index(name) for name in names], names),
        find_joints=lambda names, **kwargs: (
            [(KNEE_JOINTS + HIP_JOINTS).index(name) for name in names], names
        ),
    )
    sensor = SimpleNamespace(
        data=SimpleNamespace(current_contact_time=torch.ones(n, 2)),
        find_bodies=lambda names, **kwargs: ([FOOT_NAMES.index(name) for name in names], names),
    )
    command = Manager(
        asset=asset, command_heading_w=torch.zeros(n),
        cmd_linvel_h=torch.zeros(n, 3), cmd_linvel_b=torch.zeros(n, 3),
        is_standing_env=torch.zeros(n, 1, dtype=torch.bool),
        force_impulse=Force(n), force_drive=Force(n),
    )
    return SimpleNamespace(num_envs=n, device="cpu", step_dt=.02, physics_dt=.005,
                           command_manager=command, extra={},
                           scene=SimpleNamespace(articulations={"robot": asset},
                                                 sensors={"contact_forces": sensor}))


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


class SupportKneeTests(unittest.TestCase):
    def test_actual_lower_stop_is_costly_but_swing_and_normal_flexion_are_free(self):
        env = make_env(4)
        data = env.command_manager.asset.data
        data.joint_pos[1:, 0] = -.088
        env.scene.sensors["contact_forces"].data.current_contact_time[2:, 0] = 0.
        env.scene.sensors["contact_forces"].data.current_contact_time[3, 1] = 0.
        reward = SCOPE["facet_knee_extension_margin"](
            env, 1.5, KNEE_JOINTS, FOOT_NAMES
        )
        values, active = reward.compute()
        self.assertEqual(active.item(), 3)
        self.assertLess(values[1].item(), -1.5)
        torch.testing.assert_close(values[[0, 2, 3]], torch.zeros(3, 1))

    def test_same_support_margin_applies_to_either_force_direction_and_release(self):
        env = make_env(3)
        c = env.command_manager
        c.force_drive.value[:, 0] = torch.tensor([8., -8., 0.])
        c.is_standing_env[2] = True
        c.asset.data.joint_pos[:, 0] = -.088
        reward = SCOPE["facet_knee_extension_margin"](env, 1.5, KNEE_JOINTS, FOOT_NAMES)
        values, _ = reward.compute()
        torch.testing.assert_close(values, values[0].expand_as(values))


class ReleaseCenterTests(unittest.TestCase):
    def make_case(self, n=1):
        env = make_env(n)
        env.command_manager.is_standing_env[:] = True
        return env, SCOPE["facet_stand_support_center"](env, 1.5, FOOT_NAMES)

    def warmup(self, reward):
        for _ in range(40):
            reward.update()

    def test_normal_width_cannot_hide_shared_side_bias_and_fore_aft_is_free(self):
        env, reward = self.make_case(3)
        data = env.command_manager.asset.data
        # Latest replay: 25 cm width, but both feet displaced to the right.
        data.body_link_pos_w[1, :2, 1] = torch.tensor([.0075, -.2422])
        data.body_link_pos_w[2, :2, 1] = torch.tensor([.2422, -.0075])
        self.warmup(reward)
        values, active = reward._compute()
        self.assertTrue(active.all())
        self.assertEqual(values[0].item(), 0.)
        self.assertLess(values[1].item(), -.5)
        torch.testing.assert_close(values[1], values[2])
        data.body_link_pos_w[:, :2, 0] = torch.tensor([.4, -.1])
        torch.testing.assert_close(reward._compute()[0], values)

    def test_recovery_single_support_and_repull_are_not_centered(self):
        env, reward = self.make_case(3)
        env.command_manager.asset.data.body_link_pos_w[:, :2, 1] -= .12
        reward.update()
        self.assertFalse(reward._compute()[1].any())
        self.warmup(reward)
        env.scene.sensors["contact_forces"].data.current_contact_time[1, 1] = 0.
        env.command_manager.force_drive.value[2, 0] = -8.
        reward.update()
        values, count = reward.compute()
        self.assertEqual(count.item(), 1)
        self.assertLess(values[0].item(), 0.)
        torch.testing.assert_close(values[1:], torch.zeros(2, 1))
        self.assertEqual(reward.stand_steps[2].item(), 0.)
        reward.reset(torch.tensor([0]))
        self.assertEqual(reward.stand_steps[0].item(), 0.)

    def test_contact_ramp_and_lateral_deadband_allow_small_weight_shifts(self):
        env, reward = self.make_case(3)
        data = env.command_manager.asset.data
        data.root_com_pos_w[:, 1] = torch.tensor([.03, .12, .12])
        self.warmup(reward)
        env.scene.sensors["contact_forces"].data.current_contact_time[1] = .14
        env.scene.sensors["contact_forces"].data.current_contact_time[2] = .20
        values, active = reward._compute()
        self.assertTrue(active.all())
        self.assertEqual(values[0].item(), 0.)
        torch.testing.assert_close(values[1], .5 * values[2])

    def test_centering_uses_com_and_is_invariant_to_world_yaw_and_translation(self):
        env, reward = self.make_case()
        data = env.command_manager.asset.data
        data.root_com_pos_w[:, 1] = .12
        self.warmup(reward)
        values = reward._compute()[0]
        inverse = MATH["quat_from_yaw"](torch.tensor([-1.2]))
        shift = torch.tensor([3., -2., 1.])
        data.body_link_pos_w[:] = MATH["quat_rotate_inverse"](
            inverse.unsqueeze(1), data.body_link_pos_w
        ) + shift
        data.root_com_pos_w[:] = MATH["quat_rotate_inverse"](
            inverse, data.root_com_pos_w
        ) + shift
        data.root_link_quat_w[:] = MATH["quat_from_yaw"](torch.tensor([1.2]))
        torch.testing.assert_close(reward._compute()[0], values, atol=1e-5, rtol=1e-5)


class ReleaseLegBalanceTests(unittest.TestCase):
    def make_case(self, n=1):
        env = make_env(n)
        env.command_manager.is_standing_env[:] = True
        reward = SCOPE["facet_stand_leg_balance"](env, 1., KNEE_JOINTS, HIP_JOINTS)
        return env, reward

    def test_alternating_steps_cancel_but_persistent_pinned_knee_does_not(self):
        env, reward = self.make_case(2)
        data = env.command_manager.asset.data
        # No double-support requirement: shuffling the other foot cannot hide
        # a permanent knee difference, but phase-shifted steps are allowed.
        env.scene.sensors["contact_forces"].data.current_contact_time.zero_()
        data.joint_pos[1] = torch.tensor([-.088, .16, .18, .05])
        for i in range(80):
            swing = .2 * math.sin(2 * math.pi * i / 40)
            data.joint_pos[0] = data.default_joint_pos[0] + torch.tensor(
                [swing, -swing, -.5 * swing, .5 * swing]
            )
            reward.update()
        values, active = reward._compute()
        self.assertTrue(active.all())
        self.assertAlmostEqual(values[0].item(), 0., places=6)
        self.assertLess(values[1].item(), -.6)
        torch.testing.assert_close(reward._compute()[0], values)

    def test_force_history_full_window_and_per_environment_reset(self):
        env, reward = self.make_case(2)
        c = env.command_manager
        c.asset.data.joint_pos[:, 0] = -.088
        c.force_drive.value[:, 0] = torch.tensor([8., -8.])
        for _ in range(50):
            reward.update()
        self.assertFalse(reward._compute()[1].any())
        c.force_drive.value.zero_()
        c.asset.data.joint_pos[:] = c.asset.data.default_joint_pos
        for _ in range(39):
            reward.update()
        self.assertFalse(reward._compute()[1].any())
        reward.update()
        values, active = reward._compute()
        self.assertTrue(active.all())
        torch.testing.assert_close(values, torch.zeros(2, 1))
        reward.reset(torch.tensor([0]))
        self.assertEqual(reward.sample_count[:, 0].tolist(), [0., 40.])
        c.is_standing_env[1] = False
        reward.update()
        self.assertEqual(reward.sample_count[:, 0].tolist(), [1., 0.])
        c.is_standing_env[1] = True
        reward.update()
        torch.testing.assert_close(reward.difference_sum, torch.zeros(2, 2))


class StageConfigurationTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("hydra"), "Hydra is not installed")
    def test_hydra_inheritance_instantiates_new_rewards_only_in_stage3(self):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        with initialize_config_dir(version_base=None, config_dir=str(ROOT / "projects/facet")):
            stage3 = compose(config_name="task/facet/G1LocoFacetForceBackwardStand").task
            stage1 = compose(config_name="task/facet/G1LocoFacetForceForward").task
            stage2 = compose(config_name="task/facet/G1LocoFacetForceBackward").task
        # The policy interface and drive dynamics still come from the same
        # parent. Instantiate each new term with its fully composed arguments.
        self.assertEqual(stage3.command.linear_kd_range, stage1.command.linear_kd_range)
        self.assertEqual(stage3.input, stage1.input)
        self.assertEqual(stage3.observation, stage1.observation)
        for name in ["facet_knee_extension_margin", "facet_stand_support_center",
                     "facet_stand_leg_balance"]:
            self.assertNotIn(name, stage1.reward.loco)
            self.assertNotIn(name, stage2.reward.loco)
            cfg = OmegaConf.to_container(stage3.reward.loco[name], resolve=True)
            target = cfg.pop("_target_")
            reward = SCOPE[target](make_env(2), **cfg)
            reward.update()
            values, _ = reward.compute()
            self.assertEqual(values.shape, (2, 1))
            self.assertTrue(torch.isfinite(values).all())

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
