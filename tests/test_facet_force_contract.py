"""CPU contract regressions, without importing Isaac/Warp or the project registry.

Run in the training environment:
    python -m unittest discover -s tests -p test_facet_force_contract.py -v

Only the tested functions are loaded from their actual source AST. This avoids
requiring a simulator just to test tensor frames, command selection and GAE input.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_function(path, name, scope=None, owner=None):
    tree = ast.parse((ROOT / path).read_text())
    if owner:
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    namespace = {'torch': torch, **(scope or {})}
    code = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(path), 'exec'), namespace)
    return namespace[name]


MATH = {}
for name in ['normalize', 'quat_from_yaw', 'yaw_quat', 'quat_rotate_inverse', 'yaw_rotate', 'euler_from_quat']:
    MATH[name] = load_function('active_adaptation/utils/math.py', name, MATH)
update_gait = load_function(
    'projects/facet/facet_commands/impedance_manip.py', '_update_gait_reference',
    MATH, 'ImpedanceCommandManager',
)
advance_timer = load_function(
    'projects/facet/facet_commands/impedance_manip.py', '_advance_force_drive_timer',
    owner='ImpedanceCommandManager',
)
sample_angular = load_function(
    'projects/facet/facet_commands/impedance_manip.py', '_sample_angular_gains',
    owner='ImpedanceCommandManager',
)
set_velocity_target = load_function(
    'projects/facet/facet_commands/impedance_manip.py', '_set_velocity_target',
    MATH, 'ImpedanceCommandManager',
)
update_curriculum = load_function(
    'projects/facet/facet_commands/impedance_manip.py', 'update_curriculum',
    owner='ImpedanceCommandManager',
)
pelvis_reward = load_function(
    'projects/facet/facet_commands/rewards.py', '_compute',
    MATH, 'facet_root_pitch_roll_l2',
)
gait_phase_reward = load_function(
    'active_adaptation/envs/mdp/rewards/locomotion.py', '_compute',
    {
        '_biped_gait_phase': load_function(
            'active_adaptation/envs/mdp/rewards/locomotion.py',
            '_biped_gait_phase', {'torch': torch},
        ),
        '_command_direction_mask': load_function(
            'active_adaptation/envs/mdp/rewards/locomotion.py',
            '_command_direction_mask',
        ),
    },
    'biped_knee_phase_symmetry',
)
backward_mask = load_function('projects/facet/facet_commands/rewards.py', '_backward_command_mask')
advantage = load_function('active_adaptation/learning/ppo/ppo_base.py', 'compute_advantage', {
    'REWARD_KEY': ('next', 'reward'), 'TERM_KEY': ('next', 'terminated'), 'DONE_KEY': ('next', 'done'),
}, 'PPOBase')


class FakeTD(dict):
    def keys(self, *args):
        return super().keys()

    def set(self, key, value):
        self[key] = value


class TimedForce:
    def __init__(self):
        self.time = torch.tensor([[.98], [0.], [0.]])
        self.duration = torch.tensor([[1.], [0.], [10.]])

    @property
    def expired(self):
        return self.time >= self.duration - 1e-4


class ForceContractTest(unittest.TestCase):
    def test_velocity_setpoint_encodes_signed_speed_and_balances_damping(self):
        # Positive Kp is essential: the actor only sees x_des-x, gains and mass.
        for direction in [-1., 1.]:
            c = SimpleNamespace(
                command_heading_w=torch.tensor([0., 1., -2.]),
                cmd_linvel_b=torch.tensor([[direction * .3, 0., 0.]]).expand(3, -1),
                cmd_linvel_w=torch.zeros(3, 3), cmd_yawvel_b=torch.zeros(3, 1),
                cmd=SimpleNamespace(kp_base=torch.tensor([[12., 12.]]).expand(3, -1),
                                    kd_base=torch.tensor([[32., 6.]]).expand(3, -1)),
                asset=SimpleNamespace(data=SimpleNamespace(root_com_pos_w=torch.zeros(3, 3))),
                is_standing_env=torch.zeros(3, 1, dtype=torch.bool),
                stand_anchor_pos_w=torch.zeros(3, 3), setpoint_w=torch.zeros(3, 6),
                root_link_rpy_w=torch.zeros(3, 3),
            )
            set_velocity_target(c, torch.arange(3))
            offset = c.setpoint_w[:, :3] - c.asset.data.root_com_pos_w
            torch.testing.assert_close(offset.norm(dim=-1), torch.full((3,), .8))
            # The spring must cancel damping when measured speed equals command.
            torch.testing.assert_close(12. * offset - 32. * c.cmd_linvel_w, torch.zeros(3, 3))

    def test_curriculum_quality_is_independent_of_reward_weight(self):
        for weight in [0., 1., 1.5, 4.]:
            c = SimpleNamespace(
                velocity_curriculum=True, velocity_curriculum_tracking_threshold=.7,
                velocity_curriculum_progress=0., velocity_curriculum_extra_progress=0.,
                velocity_curriculum_step=.01, velocity_curriculum_extra_step=.01,
                device='cpu', env=SimpleNamespace(extra={}, reward_groups={
                    'loco': SimpleNamespace(funcs={'facet_track_lin_vel':
                        SimpleNamespace(enabled=True, weight=weight)})}),
            )
            update_curriculum(c, .5, .69 * weight)
            self.assertEqual(c.velocity_curriculum_progress, 0.)
            update_curriculum(c, .5, .71 * weight)
            self.assertEqual(c.velocity_curriculum_progress, .01 if weight else 0.)

    def test_pelvis_posture_reward_observes_root_even_with_upright_torso(self):
        pitches = torch.tensor([0., .1, .66])
        q = torch.zeros(3, 2, 4)
        q[:, :, 0] = 1.  # torso body 1 stays upright
        q[:, 0, 0] = torch.cos(pitches / 2.)
        q[:, 0, 2] = torch.sin(pitches / 2.)
        c = SimpleNamespace(
            asset=SimpleNamespace(data=SimpleNamespace(body_quat_w=q)),
            body_id=0, num_envs=3, target_roll=0., target_pitch=0.,
        )
        reward = pelvis_reward(c)
        self.assertEqual(tuple(reward.shape), (3, 1))
        self.assertEqual(reward[0].item(), 0.)
        self.assertAlmostEqual(reward[1].item(), -.01, places=5)
        self.assertAlmostEqual(reward[2].item(), -.4356, places=4)

    def test_huber_knee_reward_distinguishes_locked_knee_from_target_gait(self):
        asset = SimpleNamespace(data=SimpleNamespace(
            joint_pos=torch.tensor([[0.0, 0.48]]),
            default_joint_pos=torch.zeros(1, 2),
        ))
        env = SimpleNamespace(
            episode_length_buf=torch.tensor([11]), step_dt=0.02,
            command_manager=SimpleNamespace(
                is_standing_env=torch.zeros(1, 1, dtype=torch.bool),
                cmd_linvel_h=torch.tensor([[0.3, 0.0, 0.0]]),
                cmd_linvel_b=torch.tensor([[0.3, 0.0, 0.0]]),
                stand_speed_threshold=0.02,
            ),
        )
        reward_obj = SimpleNamespace(
            env=env, period=0.8, stance_ratio=0.55, asset=asset,
            knee_ids=torch.tensor([0, 1]), default=torch.zeros(1, 2),
            target_flexion=0.48, sigma=0.18, reward_type='huber',
            command_direction='forward', command_manager=env.command_manager,
        )
        good_reward, _ = gait_phase_reward(reward_obj)
        asset.data.joint_pos[0, 1] = -0.30
        locked_reward, _ = gait_phase_reward(reward_obj)
        self.assertGreater(good_reward.item(), locked_reward.item())
        self.assertLess(locked_reward.item(), 0.0)

    def manager(self, n=4, fixed=False):
        # Initial episode yaw varies; some robots have since turned by pi.
        heading = torch.linspace(-2., 2., n)
        robot_yaw = heading + torch.linspace(0., torch.pi, n)
        q = MATH['quat_from_yaw'](robot_yaw)
        local = torch.zeros(n, 3, 3)
        local[:, :, 0] = -.3
        # Transform task-frame preview into world; independent of actual velocity.
        inverse_heading = MATH['quat_from_yaw'](-heading).unsqueeze(1)
        preview = MATH['quat_rotate_inverse'](inverse_heading, local)
        return SimpleNamespace(
            fixed_velocity=fixed, command_heading_w=heading,
            asset=SimpleNamespace(data=SimpleNamespace(root_quat_w=q)),
            surr_lin_vel_target=preview, surr_yaw_vel_target=torch.zeros(n, 3, 1),
            cmd_linvel_w=torch.zeros(n, 3), cmd_linvel_b=torch.zeros(n, 3),
            cmd_linvel_h=torch.zeros(n, 3), cmd_yawvel_b=torch.zeros(n, 1),
            command_speed=torch.zeros(n, 1), stand_speed_threshold=.02,
            is_standing_env=torch.zeros(n, 1, dtype=torch.bool),
            force_release_standing=torch.zeros(n, 1, dtype=torch.bool),
            # Index zero can still be measured forward velocity; it must be ignored.
            pos_base=SimpleNamespace(vel_w=torch.ones(n, 33, 3)),
        )

    def test_backward_preview_cannot_be_relabelled_by_forward_robot_or_yaw(self):
        for n in [1, 4, 4096]:
            c = self.manager(n)
            update_gait(c)
            torch.testing.assert_close(c.cmd_linvel_h[:, 0], torch.full((n,), -.3))
            self.assertEqual(tuple(backward_mask(c).shape), (n, 1))
            self.assertTrue(backward_mask(c).all().item())

    def test_zero_preview_is_standing_even_with_forward_measured_velocity(self):
        c = self.manager()
        c.surr_lin_vel_target.zero_()
        update_gait(c)
        self.assertTrue(c.is_standing_env.all().item())
        self.assertFalse(backward_mask(c).any().item())
        torch.testing.assert_close(c.cmd_linvel_w, torch.zeros(4, 3))

    def test_fixed_velocity_samples_are_not_overwritten(self):
        c = self.manager(fixed=True)
        c.cmd_linvel_b.fill_(.2)
        c.cmd_linvel_w.fill_(.1)
        c.cmd_yawvel_b.fill_(.15)
        before = [v.clone() for v in [c.cmd_linvel_b, c.cmd_linvel_w, c.cmd_yawvel_b]]
        update_gait(c)
        for a, b in zip(before, [c.cmd_linvel_b, c.cmd_linvel_w, c.cmd_yawvel_b]):
            torch.testing.assert_close(a, b)

    def test_release_wait_starts_once_and_start_delay_counts_down(self):
        c = SimpleNamespace(
            force_drive=TimedForce(), env=SimpleNamespace(step_dt=.02),
            force_drive_wait=torch.tensor([[0.], [1.], [0.]]),
            force_drive_interval_range=(2., 2.),
        )
        advance_timer(c)
        torch.testing.assert_close(c.force_drive_wait, torch.tensor([[2.], [.98], [0.]]))
        for _ in range(10):
            advance_timer(c)
        torch.testing.assert_close(c.force_drive_wait, torch.tensor([[1.8], [.78], [0.]]))

    def test_angular_override_preserves_zero_linear_stiffness_and_other_envs(self):
        c = SimpleNamespace(
            cmd=SimpleNamespace(kp_base=torch.zeros(4, 2), kd_base=torch.full((4, 2), 4.)),
            angular_kp_range=(12., 12.), angular_kd_range=(6., 6.), device='cpu',
        )
        sample_angular(c, torch.tensor([1, 3]))
        torch.testing.assert_close(c.cmd.kp_base[:, 0], torch.zeros(4))
        torch.testing.assert_close(c.cmd.kp_base[:, 1], torch.tensor([0., 12., 0., 12.]))
        torch.testing.assert_close(c.cmd.kd_base[:, 1], torch.tensor([4., 6., 4., 6.]))

    def test_release_latch_gates_backward_even_if_preview_has_not_cleared(self):
        c = self.manager()
        c.force_release_standing.fill_(True)
        update_gait(c)
        self.assertTrue(c.is_standing_env.all().item())
        self.assertFalse(backward_mask(c).any().item())
        c.force_release_standing.zero_()
        update_gait(c)
        self.assertFalse(c.is_standing_env.any().item())
        self.assertTrue(backward_mask(c).all().item())

    def test_signed_rewards_reach_gae_and_legacy_clamp_remains_optional(self):
        observed = []
        def gae(**kwargs):
            observed.append(kwargs['reward'])
            return kwargs['reward'], kwargs['reward']
        shape = (2, 1, 1)
        td = FakeTD({
            'state_value': torch.zeros(shape), ('next', 'state_value'): torch.zeros(shape),
            ('next', 'reward'): torch.tensor([[[-2.]], [[-5.]]]),
            ('next', 'discount'): torch.ones(shape),
            ('next', 'terminated'): torch.zeros(shape, dtype=torch.bool),
            ('next', 'done'): torch.zeros(shape, dtype=torch.bool),
        })
        advantage(SimpleNamespace(gae=gae), td, None, 'adv', 'ret', clamp_reward=False)
        torch.testing.assert_close(observed[-1], td['next', 'reward'])
        advantage(SimpleNamespace(gae=gae), td, None, 'adv', 'ret', clamp_reward=True)
        torch.testing.assert_close(observed[-1], torch.zeros(shape))


if __name__ == '__main__':
    unittest.main()
