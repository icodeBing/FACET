"""CPU regression checks for FACET's PPO ratio; no simulator is needed.

Run from the repository root: python scripts/check_facet_ppo.py
"""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
from torch import nn
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import ProbabilisticActor


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_update(method_name="_update"):
    # Importing active_adaptation starts simulator-related imports. Execute the
    # actual update method in isolation, with real torch/TensorDict/actor objects.
    source = REPO_ROOT / "active_adaptation/learning/ppo/ppo_facet.py"
    tree = ast.parse(source.read_text())
    policy = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                  and node.name == "PPOPolicy")
    update = next(node for node in policy.body if isinstance(node, ast.FunctionDef)
                  and node.name == method_name)
    update.decorator_list = []  # Execute the numerical method without simulator setup.
    normalize = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "normalize")
    common_tree = ast.parse((REPO_ROOT / "active_adaptation/learning/ppo/common.py").read_text())
    common_helpers = [node for node in common_tree.body if isinstance(node, ast.FunctionDef)
                      and node.name in ("make_batch", "soft_copy_")]
    namespace = {
        "torch": torch, "nn": nn, "F": F,
        "TensorDict": TensorDict, "ACTION_KEY": "action",
        "REWARD_KEY": ("next", "reward"), "pytree": torch.utils._pytree,
        "Mod": TensorDictModule, "Seq": TensorDictSequential,
        "OBS_KEY": "policy", "OBS_PRIV_KEY": "priv",
        "IndependentNormal": independent_normal,
    }
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name in ("require_finite", "check_tensordict", "checked_tensors", "checked_step")]
    helpers += [node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == "FiniteCheck"]
    exec(compile(ast.Module(body=helpers + [normalize] + common_helpers + [update], type_ignores=[]),
                 str(source), "exec"), namespace)
    return namespace[method_name]


def independent_normal(loc, scale):
    return torch.distributions.Independent(
        torch.distributions.Normal(loc, scale.clamp_min(1e-6)), 1
    )


class ActorParameters(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Linear(5, 12)
        self.log_std = nn.Parameter(torch.zeros(12))

    def forward(self, obs):
        loc = self.mean(obs)
        return loc, self.log_std.exp().expand_as(loc)


class RatioRegression(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def make_case(self, phase, symaug, lr=0.0):
        parameters = ActorParameters()
        actor = ProbabilisticActor(
            module=TensorDictModule(parameters, ["_obs_normed"], ["loc", "scale"]),
            in_keys=["loc", "scale"], out_keys=["action"],
            distribution_class=independent_normal,
            return_log_prob=True, log_prob_key="action_log_prob",
        )
        critic = TensorDictModule(nn.Linear(5, 1), ["_obs_normed"], ["state_value"])
        encoder = TensorDictModule(nn.Linear(3, 3), ["_priv_normed"], ["_priv_feature"])
        opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters())
                               + list(encoder.parameters()), lr=lr)
        policy = SimpleNamespace(
            cfg=SimpleNamespace(phase=phase, symaug=symaug, entropy_coef=0.002, target_kl=None),
            actor_teacher=actor, actor_student=actor,
            opt_teacher=opt, opt_finetune=opt,
            encoder_priv=encoder,
            critic=critic,
            clip_param=0.2, max_grad_norm=1.0,
            critic_loss_fn=nn.MSELoss(reduction="none"),
        )
        if symaug:
            for name in ("obs_transform", "priv_transform", "ext_transform", "act_transform"):
                setattr(policy, name, torch.neg)
        td = TensorDict({
            "_obs_normed": torch.randn(256, 5),
            "_priv_normed": torch.randn(256, 3),
            "_ext_normed": torch.randn(256, 6),
            "step_count": torch.full((256, 1), 3),
            "is_init": torch.zeros(256, 1, dtype=torch.bool),
            "ret": torch.randn(256, 1),
            "adv": torch.randn(256, 1),
        }, batch_size=[256])
        with torch.no_grad():
            dist = actor.get_dist(td.copy())
            td["action"] = dist.sample()
            td["action_log_prob"] = dist.log_prob(td["action"])
            if symaug:
                symmetry = TensorDict({
                    "_obs_normed": -td["_obs_normed"],
                    "action": -td["action"],
                }, batch_size=td.batch_size)
                symmetry_dist = actor.get_dist(symmetry)
                td["sym_action_log_prob"] = symmetry_dist.log_prob(symmetry["action"])
        return policy, actor, parameters, td

    def test_unchanged_policy_has_zero_kl_and_preserves_rollout(self):
        for phase in ("train", "finetune"):
            for symaug in (False, True):
                with self.subTest(phase=phase, symaug=symaug):
                    policy, actor, _, td = self.make_case(phase, symaug)
                    action = td["action"].clone()
                    old_log_prob = td["action_log_prob"].clone()
                    old_sym_log_prob = td.get("sym_action_log_prob", None)
                    if old_sym_log_prob is not None:
                        old_sym_log_prob = old_sym_log_prob.clone()
                    info = load_update()(policy, td)
                    self.assertLess(abs(info["actor/approx_kl"].item()), 1e-6)
                    self.assertEqual(info["actor/clamp_ratio"].item(), 0.0)
                    torch.testing.assert_close(td["action"], action, rtol=0, atol=0)
                    torch.testing.assert_close(td["action_log_prob"], old_log_prob, rtol=0, atol=0)
                    if old_sym_log_prob is not None:
                        torch.testing.assert_close(
                            td["sym_action_log_prob"], old_sym_log_prob, rtol=0, atol=0
                        )
                    self.assertEqual("actor/symmetry_loss" in info, symaug)
                    self.assertTrue(all(torch.isfinite(value).all() for value in info.values()))
                    self.assertTrue(all(not value.requires_grad for value in info.values()))
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                        for p in actor.parameters()))

    def test_changed_policy_uses_stored_old_probability(self):
        for symaug in (False, True):
            with self.subTest(symaug=symaug):
                policy, actor, parameters, td = self.make_case("train", symaug)
                action = td["action"].clone()
                old_log_prob = td["action_log_prob"].clone()
                old_sym_log_prob = td.get("sym_action_log_prob", None)
                if old_sym_log_prob is not None:
                    old_sym_log_prob = old_sym_log_prob.clone()
                with torch.no_grad():
                    parameters.mean.weight.add_(0.15)
                    dist = actor.get_dist(td.copy())
                    log_ratio = dist.log_prob(action) - old_log_prob
                    if symaug:
                        symmetry = TensorDict({
                            "_obs_normed": -td["_obs_normed"],
                            "action": -action,
                        }, batch_size=td.batch_size)
                        symmetry_dist = actor.get_dist(symmetry)
                        symmetry_log_ratio = (
                            symmetry_dist.log_prob(symmetry["action"]) - old_sym_log_prob
                        )
                        log_ratio = torch.cat([log_ratio, symmetry_log_ratio])
                    ratio = log_ratio.exp()
                    expected_kl = (ratio - 1 - log_ratio).mean()
                    expected_clip = ((ratio - 1).abs() > 0.2).float().mean()
                info = load_update()(policy, td)
                self.assertGreater(expected_kl.item(), 0.01)
                torch.testing.assert_close(info["actor/approx_kl"], expected_kl)
                torch.testing.assert_close(info["actor/clamp_ratio"], expected_clip)
                torch.testing.assert_close(td["action"], action, rtol=0, atol=0)
                torch.testing.assert_close(td["action_log_prob"], old_log_prob, rtol=0, atol=0)
                if old_sym_log_prob is not None:
                    torch.testing.assert_close(
                        td["sym_action_log_prob"], old_sym_log_prob, rtol=0, atol=0
                    )

    def test_multiple_updates_preserve_rollout_and_change_parameters(self):
        policy, _, parameters, td = self.make_case("train", False, lr=5e-4)
        before = parameters.mean.weight.detach().clone()
        action = td["action"].clone()
        old_log_prob = td["action_log_prob"].clone()
        update = load_update()
        for _ in range(3):
            update(policy, td)
        self.assertFalse(torch.equal(before, parameters.mean.weight))
        torch.testing.assert_close(td["action"], action, rtol=0, atol=0)
        torch.testing.assert_close(td["action_log_prob"], old_log_prob, rtol=0, atol=0)

    def test_train_policy_prepares_fixed_mirrored_probabilities(self):
        policy, _, _, td = self.make_case("train", True)
        policy.train_in_keys = [key for key in td.keys() if key != "sym_action_log_prob"]
        expected_sym_log_prob = td["sym_action_log_prob"].clone()
        td = td.exclude("sym_action_log_prob")
        td["next", "reward"] = torch.ones(256, 1)
        policy.compute_advantage = lambda td, *args: td
        policy.cfg.ppo_epochs = 2
        policy.cfg.num_minibatches = 4
        update = load_update()
        policy._update = lambda batch: update(policy, batch)
        info = load_update("train_policy")(policy, td)
        torch.testing.assert_close(td["sym_action_log_prob"], expected_sym_log_prob)
        self.assertFalse(td["sym_action_log_prob"].requires_grad)
        self.assertLess(abs(info["actor/approx_kl"]), 1e-6)
        self.assertEqual(info["actor/clamp_ratio"], 0.0)

    def test_large_kl_skips_update_before_parameters_change(self):
        policy, _, parameters, td = self.make_case("train", False, lr=1e-4)
        policy.cfg.target_kl = 0.02
        with torch.no_grad():
            parameters.mean.weight.add_(0.3)
        before = parameters.mean.weight.detach().clone()
        step = Mock(wraps=policy.opt_teacher.step)
        policy.opt_teacher.step = step
        info = load_update()(policy, td)
        self.assertGreater(info["actor/approx_kl"].item(), 0.03)
        self.assertEqual(info["actor/update_applied"].item(), 0.0)
        step.assert_not_called()
        torch.testing.assert_close(parameters.mean.weight, before, rtol=0, atol=0)

    def test_nonfinite_minibatch_stops_before_optimizer_step(self):
        for key in ("action", "action_log_prob", "adv", "ret", "_obs_normed"):
            with self.subTest(key=key):
                policy, _, _, td = self.make_case("train", False, lr=1e-4)
                td[key].flatten()[0] = float("nan")
                policy.opt_teacher.step = Mock()
                with self.assertRaisesRegex(FloatingPointError, key):
                    load_update()(policy, td)
                policy.opt_teacher.step.assert_not_called()

    def test_overflowing_ratio_stops_before_optimizer_step(self):
        policy, _, _, td = self.make_case("train", False)
        td["action_log_prob"].fill_(-1e20)
        policy.opt_teacher.step = Mock()
        with self.assertRaisesRegex(FloatingPointError, "PPO ratio"):
            load_update()(policy, td)
        policy.opt_teacher.step.assert_not_called()

    def test_empty_valid_mask_stops_before_optimizer_step(self):
        policy, _, _, td = self.make_case("train", False)
        td["step_count"].zero_()
        policy.opt_teacher.step = Mock()
        with self.assertRaisesRegex(FloatingPointError, "no valid steps"):
            load_update()(policy, td)
        policy.opt_teacher.step.assert_not_called()

    def test_zero_return_variance_has_finite_diagnostics(self):
        policy, _, _, td = self.make_case("train", False)
        td["ret"].zero_()
        td["step_count"][1:] = 1
        info = load_update()(policy, td)
        self.assertTrue(torch.isfinite(info["critic/explained_var"]).item())

    def test_all_optimizer_groups_are_clipped_including_encoder(self):
        modules = {name: nn.Linear(3, 2) for name in ("actor", "critic", "encoder_priv")}
        opt = torch.optim.Adam([p for m in modules.values() for p in m.parameters()], lr=1e-4)
        for module in modules.values():
            for p in module.parameters():
                p.grad = torch.full_like(p, 10000.0)
        before = modules["encoder_priv"].weight.detach().clone()
        norms = load_update().__globals__["checked_step"](opt, modules, 1.0)
        for name, module in modules.items():
            self.assertGreater(norms[name].item(), 1.0)
            norm = torch.cat([p.grad.flatten() for p in module.parameters()]).norm()
            self.assertLessEqual(norm.item(), 1.00001)
        self.assertFalse(torch.equal(before, modules["encoder_priv"].weight))

    def test_nonfinite_gradient_in_any_group_prevents_step(self):
        for bad_group in ("actor", "critic", "encoder_priv", "adapt"):
            with self.subTest(group=bad_group):
                modules = {name: nn.Linear(3, 2) for name in ("actor", "critic", "encoder_priv", "adapt")}
                for name, module in modules.items():
                    for p in module.parameters():
                        p.grad = torch.full_like(p, float("inf") if name == bad_group else 1.0)
                opt = Mock()
                with self.assertRaisesRegex(FloatingPointError, bad_group):
                    load_update().__globals__["checked_step"](opt, modules, 1.0)
                opt.step.assert_not_called()

    def test_rollout_check_rejects_invalid_action(self):
        guard = load_update().__globals__["checked_tensors"]
        action = torch.zeros(4, 12)
        torch.testing.assert_close(guard(action, context="rollout action", names=["action"]), action)
        action[0, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "rollout action.*action"):
            guard(action, context="rollout action", names=["action"])

    def test_large_kl_ends_remaining_epochs(self):
        policy, _, _, td = self.make_case("train", False)
        policy.train_in_keys = list(td.keys())
        td["next", "reward"] = torch.ones(256, 1)
        policy.compute_advantage = lambda td, *args: td
        policy.cfg.ppo_epochs = 2
        policy.cfg.num_minibatches = 4
        policy._update = Mock(return_value={
            "actor/update_applied": torch.tensor(0.0), "actor/approx_kl": torch.tensor(1.0)
        })
        info = load_update("train_policy")(policy, td)
        policy._update.assert_called_once()
        self.assertEqual(info["actor/early_stop"], 1.0)

    def test_real_rollout_sequence_checks_inputs_and_actions(self):
        policy, _, _, td = self.make_case("train", False)
        policy.vecnorm = TensorDictSequential(
            TensorDictModule(nn.Identity(), ["policy"], ["_obs_normed"]),
            TensorDictModule(nn.Identity(), ["priv"], ["_priv_normed"]),
            TensorDictModule(nn.Identity(), ["ext"], ["_ext_normed"]),
        )
        raw = TensorDict({"policy": td["_obs_normed"], "priv": td["_priv_normed"],
                          "ext": td["_ext_normed"]}, batch_size=td.batch_size)
        rollout = load_update("get_rollout_policy")(policy, "eval")
        self.assertTrue(torch.isfinite(rollout(raw.clone())["action"]).all().item())
        raw["policy"][0, 0] = float("nan")
        with self.assertRaises(Exception) as raised:
            rollout(raw.clone())
        cause = raised.exception
        while not isinstance(cause, FloatingPointError) and (cause.__cause__ is not None or cause.__context__ is not None):
            cause = cause.__cause__ or cause.__context__
        self.assertIsInstance(cause, FloatingPointError)
        self.assertIn("rollout observations", str(cause))
        raw["policy"][0, 0] = 0.0
        policy.actor_teacher = TensorDictModule(
            lambda obs: torch.full((obs.shape[0], 12), float("nan")),
            ["_obs_normed"], ["action"]
        )
        rollout = load_update("get_rollout_policy")(policy, "eval")
        with self.assertRaises(Exception) as raised:
            rollout(raw.clone())
        cause = raised.exception
        while not isinstance(cause, FloatingPointError) and (cause.__cause__ is not None or cause.__context__ is not None):
            cause = cause.__cause__ or cause.__context__
        self.assertIsInstance(cause, FloatingPointError)
        self.assertIn("rollout action", str(cause))

    def test_invalid_checkpoint_is_rejected_before_loading(self):
        policy = SimpleNamespace()
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                state = {"actor_teacher": {"weight": torch.tensor([value])}, "last_phase": "train"}
                with self.assertRaisesRegex(FloatingPointError, "checkpoint.*actor_teacher/weight"):
                    load_update("load_state_dict")(policy, state)

    def test_adaptation_updates_and_stops_on_invalid_loss(self):
        policy, _, _, td = self.make_case("train", False, lr=1e-4)
        td = td.reshape(8, 32)
        policy.cfg.train_every = 32
        policy.cfg.num_minibatches = 4
        policy.encoder_priv = TensorDictSequential(
            TensorDictModule(nn.Identity(), ["_priv_normed"], ["_priv_feature"]),
            TensorDictModule(nn.Identity(), ["_ext_normed"], ["ext_feature"]),
        )
        linear = nn.Linear(5, 9)
        policy.adapt_module = TensorDictSequential(
            TensorDictModule(linear, ["_obs_normed"], ["pred"]),
            TensorDictModule(lambda pred: pred.split([3, 6], dim=-1),
                             ["pred"], ["_priv_pred", "ext_pred"]),
        )
        policy.adapt_ema = copy.deepcopy(policy.adapt_module)
        policy.adapt_loss_fn = nn.MSELoss(reduction="none")
        policy.opt_adapt = torch.optim.Adam(linear.parameters(), lr=1e-4)
        before = linear.weight.detach().clone()
        info = load_update("train_adapt")(policy, td)
        self.assertFalse(torch.equal(before, linear.weight))
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in info.values()))
        td["_ext_normed"].fill_(float("nan"))
        policy.opt_adapt.step = Mock()
        with self.assertRaisesRegex(FloatingPointError, "adapt loss"):
            load_update("train_adapt")(policy, td)
        policy.opt_adapt.step.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
