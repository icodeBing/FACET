"""Training limits and progress, independent of the simulator runtime."""

def iteration_limit(total_frames, world_size, frames_per_batch):
    """None denotes an open-ended run; positive budgets retain batch rounding."""
    if total_frames is None or total_frames == -1:
        return None
    if total_frames <= 0:
        raise ValueError("total_frames must be positive, -1, or null")
    iterations = total_frames // world_size // frames_per_batch
    if iterations < 1:
        raise ValueError("total_frames is smaller than one distributed rollout batch")
    return iterations


def training_progress(iteration, total_iters, schedule_iters):
    """An infinite run still needs a finite horizon for curricula/schedules."""
    horizon = total_iters if total_iters is not None else schedule_iters
    return min(iteration / max(horizon - 1, 1), 1.0)
