"""CPU regression for rollout -> field selection -> minibatches -> PPO updates.

Run in the training environment with:
    python -m unittest discover -s tests -p test_facet_finetune.py -v
"""

import math
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf
from torchrl.data import Composite, Unbounded

from active_adaptation.learning.modules import VecNorm
from active_adaptation.learning.ppo.ppo_facet import PPOConfig, PPOPolicy


class FacetFinetuneTest(unittest.TestCase):
    def make_policy(self, phase):
        torch.manual_seed(7)
        n = 4
        spec = Composite({
            key: Unbounded((n, dim))
            for key, dim in [("policy", 8), ("priv", 5), ("ext", 3)]
        }, shape=(n,))
        cfg = OmegaConf.create(asdict(PPOConfig(
            phase=phase, train_every=8, ppo_epochs=2,
            num_minibatches=2, target_kl=None,
        )))
        return PPOPolicy(
            cfg, spec, Composite(action=Unbounded((n, 3)), shape=(n,)),
            Unbounded((n, 1)), "cpu",
            SimpleNamespace(action_manager=SimpleNamespace(action_dim=3)),
        )

    @torch.no_grad()
    @VecNorm.freeze()
    def collect(self, policy):
        rollout = policy.get_rollout_policy(critic=True)
        n = policy.observation_spec.shape[0]
        hx = torch.randn(n, 128)
        transitions = []
        for t in range(policy.cfg.train_every):
            is_init = torch.full((n, 1), t in (0, 4), dtype=torch.bool)
            hx = hx * (~is_init)
            td = policy.observation_spec.rand()
            td["is_init"] = is_init
            td["adapt_hx"] = hx.clone()
            td["step_count"] = torch.full((n, 1), t % 4)
            rollout(td)
            next_td = policy.observation_spec.rand()
            next_td["reward"] = torch.rand(n, 1)
            next_td["discount"] = torch.ones(n, 1)
            next_td["done"] = torch.full((n, 1), t == 3, dtype=torch.bool)
            next_td["terminated"] = next_td["done"].clone()
            if policy.cfg.phase == "finetune":
                hx = td["next", "adapt_hx"].clone()
                next_td["adapt_hx"] = hx.clone()
            td["next"] = next_td
            # Match scripts/train_ppo.py: private rollout outputs are discarded.
            td = td.exclude(*[
                key for key in td.keys(True, True)
                if isinstance(key, str) and key.startswith("_")
            ])
            transitions.append(td)
        return torch.stack(transitions, dim=1)

    def test_finetune_through_train_op(self):
        policy = self.make_policy("finetune")
        teacher_before = [p.detach().clone() for p in policy.actor_teacher.parameters()]
        encoder_before = [p.detach().clone() for p in policy.encoder_priv.parameters()]
        for _ in range(2):
            data = self.collect(policy)
            action = data["action"].clone()
            log_prob = data["action_log_prob"].clone()
            student_before = [p.detach().clone() for p in policy.actor_student.parameters()]
            critic_before = [p.detach().clone() for p in policy.critic.parameters()]
            update = policy._update
            first_batch = True

            def checked_update(batch):
                nonlocal first_batch
                self.assertEqual(batch["adapt_hx"].shape[-1], 128)
                self.assertNotIn("_priv_pred", batch.keys())
                self.assertNotIn("ext_pred", batch.keys())
                if first_batch:
                    # Saved hx must still pair with its observation after shuffle.
                    with torch.no_grad():
                        expected = batch.select("_obs_normed", "is_init", "adapt_hx")
                        policy.adapt_ema(expected)
                        dist = policy.actor_student.get_dist(expected)
                        torch.testing.assert_close(
                            dist.log_prob(batch["action"]), batch["action_log_prob"],
                        )
                    first_batch = False
                return update(batch)

            with patch.object(policy, "_update", side_effect=checked_update) as spy:
                metrics = policy.train_op(data)
            self.assertEqual(spy.call_count, 4)
            self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
            self.assertEqual(metrics["actor/update_applied"], 1.0)
            self.assertIn("adapt/priv_loss", metrics)
            self.assertTrue(any(
                not torch.equal(before, after)
                for before, after in zip(student_before, policy.actor_student.parameters())
            ))
            self.assertTrue(any(
                not torch.equal(before, after)
                for before, after in zip(critic_before, policy.critic.parameters())
            ))
            torch.testing.assert_close(data["action"], action)
            torch.testing.assert_close(data["action_log_prob"], log_prob)
        for before, after in zip(teacher_before, policy.actor_teacher.parameters()):
            torch.testing.assert_close(before, after)
        for before, after in zip(encoder_before, policy.encoder_priv.parameters()):
            torch.testing.assert_close(before, after)

    def test_teacher_ppo_does_not_require_hidden_state(self):
        policy = self.make_policy("train")
        # Skip the alternating adaptation update to isolate teacher PPO.
        policy.num_updates = 1
        data = self.collect(policy).exclude("adapt_hx")
        metrics = policy.train_op(data)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        self.assertEqual(metrics["actor/update_applied"], 1.0)


if __name__ == "__main__":
    unittest.main()
