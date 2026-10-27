from typing import cast

try:
    from isaaclab.utils import configclass
except ModuleNotFoundError:
    def configclass(cls):
        return cls

from active_adaptation.assets import AssetCfg
from active_adaptation.envs.backends.mujoco.adapter import (
    MujocoSceneAdapter,
    MujocoSimAdapter,
)
from active_adaptation.envs.env_base import _EnvBase
from active_adaptation.registry import Registry


class _MujocoDebugDraw:
    """Isaac-compatible no-op debug drawer for MuJoCo rollouts.

    FACET command callbacks are shared by Isaac and MuJoCo and may emit
    vectors/points every control step.  MuJoCo's passive viewer has no
    Isaac debug-draw interface, so these calls must be harmless during
    playback rather than aborting the environment step.
    """

    def clear(self):
        pass

    def plot(self, *args, **kwargs):
        pass

    def vector(self, *args, **kwargs):
        pass

    def point(self, *args, **kwargs):
        pass


class MujocoBackendEnv(_EnvBase):
    """MuJoCo backend env: only scene/sim construction."""

    def __init__(self, cfg, device: str, headless: bool = True):
        super().__init__(cfg, device, headless)
        self.robot = self.scene.articulations["robot"]
        # Shared MDP debug callbacks expect this attribute on every backend.
        self.debug_draw = _MujocoDebugDraw()

    def setup_scene(self):
        from active_adaptation.envs.backends.mujoco.mujoco import MJScene, MJSim
        from active_adaptation.envs.terrain import TERRAINS_MUJOCO

        registry = Registry.instance()
        asset_cfg = cast(AssetCfg, registry.get("asset", self.cfg.robot.name))

        @configclass
        class SceneCfg:
            robot = asset_cfg.mujoco()
            contact_forces = "robot"
            terrain = TERRAINS_MUJOCO.get(self.cfg.terrain, TERRAINS_MUJOCO["plane"])

        scene = MJScene(SceneCfg(), headless=self.headless)
        sim = MJSim(scene, camera_cfg=self.cfg.get("camera", {}))
        self.scene = MujocoSceneAdapter(scene)
        self.sim = MujocoSimAdapter(sim)
        self.terrain_type = "plane"


__all__ = ["MujocoBackendEnv"]
