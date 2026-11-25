import torch
import hydra
import numpy as np
import wandb
import logging
import time
import datetime
import itertools
import os
import signal
from pathlib import Path
from training_runtime import iteration_limit, training_progress

from omegaconf import OmegaConf, DictConfig

from collections import OrderedDict
from tqdm import tqdm
from setproctitle import setproctitle

from torchrl.envs.utils import set_exploration_type, ExplorationType
from tensordict.nn import TensorDictModuleBase

import active_adaptation as aa
from active_adaptation.utils.profiling import ScopedTimer
from active_adaptation.learning.ppo.ppo_base import PPOBase

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


FILE_PATH = Path(__file__).resolve().parent
CONFIG_PATH = FILE_PATH.parent / "cfg"


@hydra.main(config_path=str(CONFIG_PATH), config_name="train", version_base=None)
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    aa.init(cfg, auto_rank=True)

    print(
        f"is_distributed: {aa.is_distributed()}, local_rank: {aa.get_local_rank()}/{aa.get_world_size()}"
    )

    if aa.is_main_process():
        # W&B creates its wandb/ subdirectory under dir; anchor it to this
        # script so Hydra's working directory and WANDB_DIR cannot move it.
        (FILE_PATH / "wandb").mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            dir=str(FILE_PATH),
            job_type=cfg.wandb.job_type,
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            tags=cfg.wandb.tags,
        )
        run.config.update(OmegaConf.to_container(cfg))
        run.config["world_size"] = aa.get_world_size()

        default_run_name = (
            f"{cfg.exp_name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        # run_idx = run.name.split("-")[-1]
        run_idx = (run.name or run.id or "local").split("-")[-1]
        run.name = f"{run_idx}-{default_run_name}"
        setproctitle(run.name)

        # Disabled W&B runs may report a /tmp run.dir even with dir set.
        # Choose the checkpoint directory ourselves in every W&B mode.
        run_dir = FILE_PATH / "wandb" / (
            f"run-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}-{run.id}"
        ) / "files"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpoint directory: {run_dir}")
        cfg_save_path = run_dir / "cfg.yaml"
        OmegaConf.save(cfg, cfg_save_path)
        run.save(str(cfg_save_path), policy="now", base_path=str(run_dir))

    from active_adaptation.helpers import make_env_policy, evaluate
    from active_adaptation.utils.helpers import EpisodeStats

    env, policy = make_env_policy(cfg)
    policy: PPOBase

    frames_per_batch = env.num_envs * cfg.algo.train_every
    total_iters = iteration_limit(
        cfg.get("total_frames", -1), aa.get_world_size(), frames_per_batch
    )
    schedule_iters = total_iters
    if total_iters is None:
        schedule_iters = iteration_limit(
            cfg.get("schedule_frames", 150_000_000), aa.get_world_size(), frames_per_batch
        )
        if schedule_iters is None:
            raise ValueError("schedule_frames must be a positive finite frame budget")
        logging.info("Unlimited training: stop with Ctrl+C or SIGTERM; schedule horizon=%s updates", schedule_iters)
    
    checkpoint_interval = cfg.checkpoint_interval
    upload_interval = cfg.upload_interval

    log_interval = (env.max_episode_length // cfg.algo.train_every) + 1
    logging.info(f"Log interval: {log_interval} steps")

    stats_keys = [
        k
        for k in env.reward_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys, device=env.device)

    def save(policy, checkpoint_name: str, *, upload_to_wandb: bool = True):
        ckpt_path = run_dir / f"{checkpoint_name}.pt"
        state_dict = OrderedDict()
        state_dict["wandb"] = {"name": run.name, "id": run.id}
        state_dict["policy"] = policy.state_dict()
        
        # Readers must never see a partially overwritten checkpoint_temp.pt.
        temporary_path = ckpt_path.with_suffix(".pt.tmp")
        torch.save(state_dict, temporary_path)
        os.replace(temporary_path, ckpt_path)
        if upload_to_wandb:
            run.save(str(ckpt_path), policy="now", base_path=str(run_dir))
        
        latest_link = run_dir / "checkpoint_latest.pt"
        temporary_link = run_dir / ".checkpoint_latest.tmp"
        temporary_link.unlink(missing_ok=True)
        temporary_link.symlink_to(ckpt_path.name)
        os.replace(temporary_link, latest_link)
        logging.info(f"Saved checkpoint to {ckpt_path}" + (" (wandb)" if upload_to_wandb else ""))
        return str(ckpt_path)

    assert env.training

    def should_save(i):
        if not aa.is_main_process():
            return False
        return i % checkpoint_interval == 0 or i % upload_interval == 0

    ckpt_path = None
    carry = env.reset()
    observation_keys = list(env.observation_spec.keys(True, True))
    transitions = cfg.algo.get("store_transitions", True)

    @torch.no_grad()
    @set_exploration_type(ExplorationType.RANDOM)
    def collect(carry, rollout_policy: TensorDictModuleBase, transitions: bool=False):
        """
        If transitions is True, we collect and store transitions.
        If transitions is False, we store only the trajectories. Due to autoreset, we do not have
        the next states at terminal states. In this case, we approximate V_{t+1} with V_t.
        """
        data = []
        for _ in range(cfg.algo.train_every):
            carry = rollout_policy(carry)
            td, carry = env.step_and_maybe_reset(carry)
            private_keys = [
                key
                for key in td.keys(True, True)
                if isinstance(key, str) and key.startswith("_")
            ]
            td = td.exclude(*private_keys)
            if not transitions:
                td["next"] = td["next"].exclude(*observation_keys)

            data.append(td.to(policy.device))
        data = torch.stack(data, dim=1)

        if not transitions:
            state_value = data["state_value"]
            next_state_value = policy.compute_value(carry.copy())["state_value"]
            next_state_value = torch.cat([
                data["state_value"][:, 1:],
                next_state_value.unsqueeze(1),
            ], dim=1)
            # since we have not stored the terminal states, we approximate V_{t+1} with V_t
            data["next", "state_value"] = torch.where(
                data["next", "done"],
                state_value,
                next_state_value,
            )
        return data, carry

    env_frames = 0

    if hasattr(policy.cfg, "stages"):
        stages = policy.cfg.stages
    else:
        stages = ("",)

    if total_iters is None and len(stages) > 1:
        raise ValueError("Unlimited training requires a single algorithm stage")

    stop_requested = False

    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    previous_handlers = {
        sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    stop_flag = torch.zeros(1, dtype=torch.int32, device=env.device)

    for stage in stages:

        policy.on_stage_start(stage)
        rollout_policy = policy.get_rollout_policy(
            "train",
            critic=not transitions,
        )

        iterations = itertools.count() if total_iters is None else range(total_iters)
        if aa.is_main_process():
            progress = tqdm(iterations, total=total_iters, desc=stage)
        else:
            progress = iterations

        tracking_score = None
        for i in progress:
            stop_flag.fill_(int(stop_requested))
            if aa.get_world_size() > 1:
                torch.distributed.all_reduce(stop_flag, op=torch.distributed.ReduceOp.MAX)
            if stop_flag.item():
                stop_requested = True
                logging.info("Stop requested; saving the last completed policy update")
                break
            rollout_start = time.perf_counter()
            with ScopedTimer("rollout") as rollout_timer:
                try:
                    data, carry = collect(carry, rollout_policy, transitions)
                except Exception:
                    logging.exception(
                        "Rollout failed at iteration %s, env_frames=%s. Last saved checkpoint: %s",
                        i, env_frames * aa.get_world_size(), ckpt_path,
                    )
                    raise
            rollout_time = rollout_timer.last_time

            episode_stats.add(data)
            env_frames += data.numel()

            info = {}
            if i % log_interval == 0 and len(episode_stats):
                for k, v in sorted(episode_stats.pop().items(True, True)):
                    key = "train/" + ("/".join(k) if isinstance(k, tuple) else k)
                    info[key] = torch.mean(v.float()).item()

            with ScopedTimer("training") as training_timer:
                try:
                    info.update(policy.train_op(data))
                except Exception:
                    logging.exception(
                        "Policy update failed at iteration %s, env_frames=%s. Last saved checkpoint: %s",
                        i, env_frames * aa.get_world_size(), ckpt_path,
                    )
                    raise
            training_time = training_timer.last_time

            info.update(env.extra)
            info.update(env.stats_ema)  # step-wise exponential moving average of stats

            tracking_score = info.get("reward.loco/facet_track_lin_vel")
            schedule_progress = training_progress(i, total_iters, schedule_iters)
            env.set_progress(schedule_progress, tracking_score)
            info.update(env.extra)

            if hasattr(policy, "step_schedule"):
                policy.step_schedule(
                    schedule_progress if total_iters is None else i / total_iters
                )

            info["env_frames"] = env_frames * aa.get_world_size()
            info["performance/rollout_fps"] = (
                data.numel() / rollout_time * aa.get_world_size()
            )
            info["performance/rollout_time"] = rollout_time
            info["performance/training_time"] = training_time
            info["performance/iter_time"] = time.perf_counter() - rollout_start

            # ``env.extra`` stores curriculum diagnostics as scalar tensors.
            # Convert those values before the console/YAML print below; the
            # old ``isinstance(v, (float, int))`` filter silently omitted them
            # from ``2>&1 | tee`` logs even though the environment updated them.
            for key, value in list(info.items()):
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    info[key] = value.detach().cpu().item()

            if should_save(i):
                should_upload = i % upload_interval == 0
                checkpoint_name = f"checkpoint_{i}" if should_upload else "checkpoint_temp"
                ckpt_path = save(policy, checkpoint_name, upload_to_wandb=should_upload)

            if aa.is_main_process():
                ScopedTimer.print_summary(clear=True, depth=3)
                print(
                    OmegaConf.to_yaml(
                        {k: v for k, v in info.items() if isinstance(v, (float, int))}
                    )
                )
                print(f"Latest checkpoint: {ckpt_path}")
                run.log(info)

        if stop_requested:
            break

    if aa.is_main_process():
        ckpt_path = save(policy, "checkpoint_final")
        if not stop_requested:
            policy_eval = policy.get_rollout_policy("eval")
            with ScopedTimer("rollout"):
                info, trajs, stats = evaluate(
                    env, policy_eval, render=cfg.eval_render, seed=cfg.seed
                )
            info["env_frames"] = env_frames
            run.log(info)
        wandb.finish()
        print(f"Final checkpoint: {ckpt_path}")
    for sig, handler in previous_handlers.items():
        signal.signal(sig, handler)
    env.close()
    exit(0)


if __name__ == "__main__":
    main()
