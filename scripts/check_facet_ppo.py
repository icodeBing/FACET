"""CPU regression checks for FACET's PPO ratio; no simulator is needed.

Run from the repository root: python scripts/check_facet_ppo.py
"""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
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
    normalize = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "normalize")
    common_tree = ast.parse((REPO_ROOT / "active_adaptation/learning/ppo/common.py").read_text())
    make_batch = next(node for node in common_tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "make_batch")
    namespace = {
        "torch": torch, "nn": nn, "F": F,
        "TensorDict": TensorDict, "ACTION_KEY": "action",
        "REWARD_KEY": ("next", "reward"), "pytree": torch.utils._pytree,
        "IndependentNormal": independent_normal,
    }
    exec(compile(ast.Module(body=[normalize, make_batch, update], type_ignores=[]),
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
        opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
        policy = SimpleNamespace(
            cfg=SimpleNamespace(phase=phase, symaug=symaug, entropy_coef=0.002),
            actor_teacher=actor, actor_student=actor,
            opt_teacher=opt, opt_finetune=opt,
            encoder_priv=lambda td: td, critic=critic,
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
