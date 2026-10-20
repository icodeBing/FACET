# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import functools
import einops
import copy
import torch.utils._pytree as pytree

from torchrl.data import Composite, TensorSpec, Unbounded
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import TensorDictPrimer
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModuleBase, 
    TensorDictModule as Mod, 
    TensorDictSequential as Seq
)
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Union, Tuple
from collections import OrderedDict

from active_adaptation.learning.modules import IndependentNormal, VecNorm
from active_adaptation.learning.modules.rnn import set_recurrent_mode, recurrent_mode
from .common import *
from .ppo_base import PPOBase



@dataclass
class PPOConfig:
    _target_: str = "active_adaptation.learning.ppo.ppo_facet.PPOPolicy"
    name: str = "ppo_facet"
    train_every: int = 32
    ppo_epochs: int = 2
    num_minibatches: int = 4
    lr: float = 1e-4
    clip_param: float = 0.2
    entropy_coef: float = 0.002
    target_kl: Union[float, None] = 0.02

    compile: bool = False # slightly improve speed on some devices

    # Optional symmetry augmentation; disabled by default for FACET training.
    symaug: bool = False
    phase: str = "train"
    in_keys: Tuple[str, ...] = (OBS_KEY, OBS_PRIV_KEY, "ext", "ext_")

cs = ConfigStore.instance()
cs.store("ppo_facet_train", node=PPOConfig(phase="train"), group="algo")
# there is no symmetry transform for RNN states (adapt_hx)
# so we don't use symmetry augmentation in adapt and finetune phases
cs.store("ppo_facet_adapt", node=PPOConfig(phase="adapt", symaug=False), group="algo")
cs.store("ppo_facet_finetune", node=PPOConfig(phase="finetune", symaug=False), group="algo")


def require_finite(context, **tensors):
    """Check on device, synchronizing only once unless an error needs details."""
    if not tensors:
        return
    checks = [torch.isfinite(value).all() for value in tensors.values()]
    finite = torch.stack([check.to(checks[0].device) for check in checks])
    if not finite.all().item():
        bad = [name for name, ok in zip(tensors, finite.tolist()) if not ok]
        raise FloatingPointError(f"FACET {context}: NaN/Inf in {', '.join(bad)}")


def check_tensordict(tensordict, context):
    require_finite(context, **{
        str(key): value for key, value in tensordict.items(True, True)
        if torch.is_tensor(value) and value.is_floating_point()
    })


def checked_tensors(*values, context, names):
    require_finite(context, **dict(zip(names, values)))
    return values if len(values) > 1 else values[0]


class FiniteCheck(nn.Module):
    def __init__(self, names, context):
        super().__init__()
        self.names = names
        self.context = context

    def forward(self, *values):
        return checked_tensors(*values, context=self.context, names=self.names)


def checked_step(optimizer, modules, max_grad_norm):
    """Check and clip every optimized module before changing any parameters."""
    norms = {}
    for name, module in modules.items():
        try:
            norms[name] = nn.utils.clip_grad_norm_(
                module.parameters(), max_grad_norm, error_if_nonfinite=True
            )
        except RuntimeError as exc:
            raise FloatingPointError(f"FACET backward: non-finite {name} gradient norm") from exc
    optimizer.step()
    require_finite("optimizer step", **{
        f"{name}/{key}": value
        for name, module in modules.items() for key, value in module.named_parameters()
    })
    return norms


class GRU(nn.Module):
    def __init__(
        self, 
        input_size, 
        hidden_size, 
        burn_in: bool = False
    ) -> None:
        super().__init__()
        self.gru = nn.GRUCell(input_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.burn_in = burn_in

    def forward(self, x: torch.Tensor, is_init: torch.Tensor, hx: torch.Tensor):
        if recurrent_mode():
            N, T = x.shape[:2]
            hx = hx[:, 0]
            output = []
            reset = 1. - is_init.float().reshape(N, T, 1)
            for i, x_t, reset_t in zip(range(T), x.unbind(1), reset.unbind(1)):
                hx = self.gru(x_t, hx * reset_t)
                if self.burn_in and i < T // 4:
                    hx = hx.detach()
                output.append(hx)
            output = torch.stack(output, dim=1)
            output = self.ln(output)
            return output, einops.repeat(hx, "b h -> b t h", t=T)
        else:
            N = x.shape[0]
            hx = self.gru(x, hx)
            output = self.ln(hx)
            return output, hx


class GRUModule(nn.Module):
    def __init__(self, dim: int, split):
        super().__init__()
        self.split = split
        self.mlp = make_mlp([128, 128])
        self.gru = GRU(128, hidden_size=128)
        self.out = nn.LazyLinear(dim)
    
    def forward(self, x, is_init, hx):
        out1 = self.mlp(x)
        out2, hx = self.gru(out1, is_init, hx)
        out3 = self.out(out2 + out1)
        if self.split is None:
            out = (out3,)
        else:
            out = torch.split(out3, self.split, dim=-1)
        return out + (hx.contiguous(),)


class PPOPolicy(PPOBase):

    train_in_keys = [
        "_obs_normed", "_priv_normed", "_ext_normed",
        ACTION_KEY, "action_log_prob",
        "adv", "ret", "is_init", "step_count"
    ]
    
    def __init__(
        self, 
        cfg: PPOConfig, 
        observation_spec: Composite, 
        action_spec: Composite, 
        reward_spec: TensorSpec,
        device,
        env
    ):
        super().__init__()
        self.cfg = PPOConfig(**cfg)
        self.device = device
        self.observation_spec = observation_spec
        assert self.cfg.phase in ["train", "adapt", "finetune"]

        self.max_grad_norm = 1.0
        self.clip_param = self.cfg.clip_param
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.adapt_loss_fn = nn.MSELoss(reduction="none")
        self.gae = GAE(0.99, 0.95)
        
        fake_input = observation_spec.zero()

        obs_dim = fake_input[OBS_KEY].shape[-1]
        priv_dim = fake_input[OBS_PRIV_KEY].shape[-1]
        ext_dim = fake_input["ext"].shape[-1]
        self.action_dim = env.action_manager.action_dim
        
        if self.cfg.symaug:
            self.obs_transform = env.observation_funcs[OBS_KEY].symmetry_transform().to(self.device)
            self.priv_transform = env.observation_funcs[OBS_PRIV_KEY].symmetry_transform().to(self.device)
            self.ext_transform = env.observation_funcs["ext"].symmetry_transform().to(self.device)
            self.act_transform = env.action_manager.symmetry_transform().to(self.device)
        
        self.vecnorm = Seq(
            Mod(VecNorm(obs_dim, obs_dim, decay=1.0), [OBS_KEY], ["_obs_normed"]),
            Mod(VecNorm(priv_dim, priv_dim, decay=1.0), [OBS_PRIV_KEY], ["_priv_normed"]),
            Mod(VecNorm(ext_dim, ext_dim, decay=1.0), ["ext"], ["_ext_normed"]),
        ).to(self.device)

        self.encoder_priv = Seq(
            Mod(nn.Sequential(make_mlp([128]), nn.LazyLinear(128)), ["_priv_normed"], ["_priv_feature"]),
            Mod(nn.Sequential(make_mlp([32]), nn.LazyLinear(32)), ["_ext_normed"], ["ext_feature"]),
        ).to(self.device)

        ext_dim = observation_spec["ext"].shape[-1]
        self.adapt_module =  Mod(
            GRUModule(128 + 32 + ext_dim, split=[128, 32, ext_dim]), 
            ["_obs_normed", "is_init", "adapt_hx"], 
            ["_priv_pred", "ext_pred", ("info", "ext_rec"), ("next", "adapt_hx")]
        ).to(self.device)
        
        def make_actor(in_keys) -> ProbabilisticActor:
            return ProbabilisticActor(
                module=Seq(
                    CatTensors(in_keys, "_actor_inp", del_keys=False, sort=False),
                    Mod(make_mlp([256, 256, 256]), ["_actor_inp"], ["_actor_feature"]),
                    Mod(Actor(self.action_dim), ["_actor_feature"], ["loc", "scale"]),
                ),
                in_keys=["loc", "scale"],
                out_keys=[ACTION_KEY],
                distribution_class=IndependentNormal,
                return_log_prob=True
            ).to(self.device)
        
        teacher_in_keys = ["_obs_normed", "_priv_feature", "ext_feature"]
        self.actor_teacher = make_actor(teacher_in_keys)
        
        student_in_keys = ["_obs_normed", "_priv_pred", "ext_pred"]
        self.actor_student = make_actor(student_in_keys)
        
        critic_in_keys = ["_obs_normed", "_priv_normed", "_ext_normed"]
        critic_mlp = nn.Sequential(make_mlp([512, 256, 128]), nn.LazyLinear(1))
        self.critic = Seq(
            CatTensors(critic_in_keys, "_critic_input", del_keys=False),
            Mod(critic_mlp, ["_critic_input"], ["state_value"])
        ).to(self.device)

        with torch.device(self.device):
            fake_input["is_init"] = torch.ones(fake_input.shape[0], 1, dtype=torch.bool)
            fake_input["adapt_hx"] = torch.zeros(fake_input.shape[0], 128)

        self.vecnorm(fake_input)
        self.encoder_priv(fake_input)
        self.actor_teacher(fake_input)
        self.adapt_module(fake_input)
        self.actor_student(fake_input)
        self.critic(fake_input)

        self.adapt_ema = copy.deepcopy(self.adapt_module)
        self.adapt_ema.requires_grad_(False)

        self.opt_teacher = torch.optim.Adam(
            [
                {"params": self.actor_teacher.parameters()},
                {"params": self.critic.parameters()},
                {"params": self.encoder_priv.parameters()},
            ],
            lr=cfg.lr
        )

        self.opt_adapt = torch.optim.Adam(
            [
                {"params": self.adapt_module.parameters()},
            ],
            lr=cfg.lr
        )

        self.opt_finetune = torch.optim.Adam(
            [
                {"params": self.actor_student.parameters()},
                {"params": self.critic.parameters()},
            ],
            lr=cfg.lr
        )
        
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        
        self.actor_teacher.apply(init_)
        # self.actor_student.apply(init_) # no need to initialize the student actor
        self.critic.apply(init_)
        self.encoder_priv.apply(init_)
        self.adapt_module.apply(init_)

        self.update = self._update
        if self.cfg.compile:
            self.update = torch.compile(self.update)
        self.num_updates = 0
    
    def make_tensordict_primer(self):
        # initialize the input for recurrent policies
        num_envs = self.observation_spec.shape[0]
        spec = Unbounded((num_envs, 128), device=self.device)
        return TensorDictPrimer({"adapt_hx": spec}, reset_key="done", expand_specs=False)

    def get_rollout_policy(self, mode: str="train", critic: bool = False):
        observation_keys = [OBS_KEY, OBS_PRIV_KEY, "ext"]
        modules = []
        if mode != "deploy":
            modules.append(Mod(
                FiniteCheck(observation_keys, "rollout observations"),
                observation_keys, observation_keys,
            ))
        modules.append(self.vecnorm)
        
        if self.cfg.phase == "train":
            modules.append(self.encoder_priv)
            modules.append(self.actor_teacher)
            # modules.append(self.adapt_module) # not used in train phase
        elif self.cfg.phase == "adapt": # this phase can be skipped
            modules.append(self.adapt_module)
            modules.append(self.actor_student)
        elif self.cfg.phase == "finetune":
            # modules.append(self.adapt_module) # we use its EMA for better stability
            modules.append(self.adapt_ema)
            modules.append(self.actor_student)
        
        if critic:
            modules.append(self.critic)

        # Stop before an invalid action reaches physics, including during play.
        if mode != "deploy":
            modules.append(Mod(
                FiniteCheck([ACTION_KEY], "rollout action"), [ACTION_KEY], [ACTION_KEY]
            ))

        policy = Seq(*modules)
        return policy

    def compute_value(self, tensordict):
        self.vecnorm(tensordict)
        return self.critic(tensordict)

    @VecNorm.freeze()
    def train_op(self, tensordict: TensorDict):
        assert VecNorm.FROZEN, "VecNorm must be frozen before training"
        info = {}
        tensordict = tensordict.exclude(("next", "stats"))
        check_tensordict(tensordict, "collected transitions")
        
        # apply vecnorm once
        self.vecnorm(tensordict)
        self.vecnorm(tensordict["next"])

        if self.cfg.phase == "train":
            info.update(self.train_policy(tensordict.copy()))
            if self.num_updates % 2 == 0:
                info.update(self.train_adapt(tensordict.copy()))
        elif self.cfg.phase == "adapt":
            info.update(self.train_adapt(tensordict.copy()))
        elif self.cfg.phase == "finetune":
            info.update(self.train_policy(tensordict.copy()))
            info.update(self.train_adapt(tensordict.copy()))
        self.num_updates += 1
        return info
    
    def train_policy(self, tensordict: TensorDict):    
        infos = []
        
        tensordict = self.compute_advantage(tensordict, self.critic, "adv", "ret")
        require_finite("GAE", adv=tensordict["adv"], ret=tensordict["ret"])

        reward = tensordict[REWARD_KEY].sum(-1)
        tensordict["adv"] = normalize(tensordict["adv"], subtract_mean=True)
        train_in_keys = list(self.train_in_keys)
        if self.cfg.symaug:
            # The mirrored sample needs its own probability under the rollout
            # policy. It is generally not equal to the original sample's
            # probability until the policy has learned exact equivariance.
            with torch.no_grad():
                symmetry = tensordict.empty()
                symmetry["_obs_normed"] = self.obs_transform(tensordict["_obs_normed"])
                symmetry["_priv_normed"] = self.priv_transform(tensordict["_priv_normed"])
                symmetry["_ext_normed"] = self.ext_transform(tensordict["_ext_normed"])
                symmetry[ACTION_KEY] = self.act_transform(tensordict[ACTION_KEY])
                self.encoder_priv(symmetry)
                old_symmetry_dist = self.actor_teacher.get_dist(symmetry)
                tensordict["sym_action_log_prob"] = old_symmetry_dist.log_prob(
                    symmetry[ACTION_KEY]
                )
            train_in_keys.append("sym_action_log_prob")
        tensordict = tensordict.select(*train_in_keys)

        early_stop = False
        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                info = self._update(minibatch)
                infos.append(info)
                if not info["actor/update_applied"].item():
                    early_stop = True
                    break
            if early_stop:
                break

        infos = pytree.tree_map(lambda *xs: sum(xs).item() / len(xs), *infos)
        infos["critic/value_mean"] = tensordict["ret"].mean().item()
        infos["critic/value_std"] = tensordict["ret"].std().item()
        infos["critic/neg_rew_ratio"] = (reward <= 0.).float().mean().item()
        infos["actor/early_stop"] = float(early_stop)
        opt = self.opt_teacher if self.cfg.phase == "train" else self.opt_finetune
        infos["actor/lr"] = opt.param_groups[0]["lr"]
        return dict(sorted(infos.items()))
    
    @set_recurrent_mode(True)
    def train_adapt(self, tensordict: TensorDict):
        infos = []

        with torch.no_grad():
            self.encoder_priv(tensordict)

        for epoch in range(2):
            for minibatch in make_batch(tensordict, self.cfg.num_minibatches, self.cfg.train_every):
                self.adapt_module(minibatch)
                priv_loss = self.adapt_loss_fn(minibatch["_priv_pred"], minibatch["_priv_feature"])
                priv_loss = (priv_loss * (~minibatch["is_init"])).mean()
                ext_loss = self.adapt_loss_fn(minibatch["ext_pred"], minibatch["ext_feature"])
                ext_loss = (ext_loss * (~minibatch["is_init"])).mean()
                loss = priv_loss + ext_loss
                require_finite("adapt loss", loss=loss)
                self.opt_adapt.zero_grad(set_to_none=True)
                loss.backward()
                norms = checked_step(self.opt_adapt, {"adapt": self.adapt_module}, self.max_grad_norm)
                infos.append(TensorDict({
                    "adapt/priv_loss": priv_loss.detach(),
                    "adapt/ext_loss": ext_loss.detach(),
                    "adapt/grad_norm": norms["adapt"],
                }, []))
        
        soft_copy_(self.adapt_module, self.adapt_ema, 0.04)
        
        infos = {k: v.mean().item() for k, v in sorted(torch.stack(infos).items())}
        return infos

    # @torch.compile
    def _update(self, tensordict: TensorDict):
        check_tensordict(tensordict, "PPO minibatch")
        original_bsize = tensordict.shape[0]
        if self.cfg.symaug:
            symmetry = tensordict.empty()
            symmetry["_obs_normed"] = self.obs_transform(tensordict["_obs_normed"])
            symmetry["_priv_normed"] = self.priv_transform(tensordict["_priv_normed"])
            symmetry["_ext_normed"] = self.ext_transform(tensordict["_ext_normed"])
            symmetry[ACTION_KEY] = self.act_transform(tensordict[ACTION_KEY])
            symmetry["action_log_prob"] = tensordict["sym_action_log_prob"]
            symmetry["is_init"] = tensordict["is_init"]
            symmetry["step_count"] = tensordict["step_count"]
            symmetry["ret"] = tensordict["ret"]
            symmetry["adv"] = tensordict["adv"]
            tensordict = torch.cat([
                tensordict.exclude("sym_action_log_prob"), symmetry
            ], dim=0)
        
        valid = (tensordict["step_count"] > 1)
        valid_cnt = valid.sum()
        if valid_cnt.item() == 0:
            raise FloatingPointError("FACET PPO minibatch: no valid steps (step_count > 1)")

        # Keep the rollout data fixed: actor.forward() writes a new action and
        # action_log_prob, which must not replace the denominator of PPO's ratio.
        action = tensordict[ACTION_KEY].detach()
        old_log_probs = tensordict["action_log_prob"].detach()
        if self.cfg.phase == "train":
            actor = self.actor_teacher
            opt = self.opt_teacher
            self.encoder_priv(tensordict)
        elif self.cfg.phase == "finetune":
            actor = self.actor_student
            opt = self.opt_finetune

        # Evaluate the collected action without sampling or overwriting it.
        dist = actor.get_dist(tensordict)
        require_finite("actor distribution", loc=tensordict["loc"], scale=tensordict["scale"])
        log_probs = dist.log_prob(action)
        entropy = (dist.entropy().reshape_as(valid) * valid).sum() / valid_cnt

        adv = tensordict["adv"]
        log_ratio = (log_probs - old_log_probs).unsqueeze(-1)
        ratio = torch.exp(log_ratio)
        require_finite("PPO ratio", log_ratio=log_ratio, ratio=ratio)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - (torch.min(surr1, surr2).reshape_as(valid) * valid).sum() / valid_cnt
        entropy_loss = - self.cfg.entropy_coef * entropy

        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(b_returns, values)
        value_loss = (value_loss.reshape_as(valid) * valid).sum() / valid_cnt
        
        loss = policy_loss + entropy_loss + value_loss
        require_finite("PPO loss", policy_loss=policy_loss, entropy_loss=entropy_loss, value_loss=value_loss, total_loss=loss)

        with torch.no_grad():
            # Use the same valid samples as the objective. A large KL ends this
            # rollout's PPO epochs before applying another oversized update.
            approx_kl = (((ratio - 1) - log_ratio) * valid).sum() / valid_cnt
            clipfrac = ((((ratio - 1).abs() > self.clip_param) * valid).sum() / valid_cnt)
            return_var = b_returns[valid].var(unbiased=False).clamp_min(1e-8)
            explained_var = 1 - value_loss.detach() / return_var
        require_finite("PPO diagnostics", approx_kl=approx_kl, explained_var=explained_var)
        update_applied = self.cfg.target_kl is None or approx_kl.item() <= 1.5 * self.cfg.target_kl

        optimized_modules = {"actor": actor, "critic": self.critic}
        if self.cfg.phase == "train":
            optimized_modules["encoder_priv"] = self.encoder_priv
        norms = {name: loss.new_zeros(()) for name in optimized_modules}

        opt.zero_grad(set_to_none=True)
        if update_applied:
            loss.backward()
            norms = checked_step(opt, optimized_modules, self.max_grad_norm)
        info = {
            "actor/policy_loss": policy_loss.detach(),
            "actor/entropy": entropy.detach(),
            "actor/grad_norm": norms["actor"],
            "actor/update_applied": loss.new_tensor(float(update_applied)),
            'actor/approx_kl': approx_kl,
            "actor/clamp_ratio": clipfrac.detach(),
            "critic/value_loss": value_loss.detach(),
            "critic/grad_norm": norms["critic"],
            "critic/explained_var": explained_var,
        }
        if self.cfg.phase == "train":
            info["encoder_priv/grad_norm"] = norms["encoder_priv"]
        if self.cfg.symaug:
            with torch.no_grad():
                info["actor/symmetry_loss"] = F.mse_loss(
                    tensordict["loc"][original_bsize:],
                    self.act_transform(tensordict["loc"][:original_bsize])
                )
        require_finite("PPO metrics", **info)
        return info
    
    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["last_phase"] = self.cfg.phase
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        require_finite("checkpoint", **{
            f"{name}/{key}": value
            for name, values in state_dict.items() if isinstance(values, dict)
            for key, value in values.items() if torch.is_tensor(value)
        })
        failed_keys = super().load_state_dict(state_dict, strict)
        if state_dict.get("last_phase", "train") == "train":
            # only copy to initialize the actor once
            # hard_copy_(self.actor, self.actor_adapt)
            hard_copy_(self.actor_teacher, self.actor_student)
        return failed_keys


def normalize(x: torch.Tensor, subtract_mean: bool=False):
    if subtract_mean:
        return (x - x.mean()) / x.std().clamp(1e-7)
    else:
        return x  / x.std().clamp(1e-7)
