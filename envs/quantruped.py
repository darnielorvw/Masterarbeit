import os
from typing import Tuple

import jax
import mujoco
from brax import base, math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from jax import numpy as jp


class Quantruped(PipelineEnv):
    """Goal-conditioned quadruped locomotion env for decentralized CRL training.

    Same skeleton as scaling-crl's Ant (four legs, target body appended as two
    unactuated slide joints), but additionally carries the last applied action
    in `state.info["last_action"]` so it can be included in the observation --
    needed for the decentralized per-leg controllers (see leg_topology.py),
    which mirror the local+neighbor observation scope of the original
    QuantrupedMultiEnv_Local environment.
    """

    def __init__(
        self,
        ctrl_cost_weight=0.5,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        reset_noise_scale=0.1,
        backend="positional",
        **kwargs,
    ):
        path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "assets", "quantruped.xml"
        )
        sys = mjcf.load(path)

        n_frames = 5

        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.005})
            n_frames = 10

        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )

        if backend == "positional":
            sys = sys.replace(
                actuator=sys.actuator.replace(
                    gear=200 * jp.ones_like(sys.actuator.gear)
                )
            )

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        # set the target q, qd (target is the last body, with 2 unactuated slide joints)
        rng, target = self._random_target(rng)
        q = q.at[-2:].set(target)
        qd = qd.at[-2:].set(0)

        pipeline_state = self.pipeline_init(q, qd)
        last_action = jp.zeros(self.sys.act_size())
        obs = self._get_obs(pipeline_state, last_action)

        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_forward": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "x_position": zero,
            "y_position": zero,
            "dist_to_goal": zero,
            "success": zero,
        }
        info = {"seed": 0, "last_action": last_action}
        state = State(pipeline_state, obs, reward, done, metrics)
        state.info.update(info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        if "steps" in state.info.keys():
            seed = state.info["seed"] + jp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]
        info = {"seed": seed, "last_action": action}

        target = pipeline_state.x.pos[-1][:2]
        dist_before = math.safe_norm(pipeline_state0.x.pos[0][:2] - target)
        dist_after = math.safe_norm(pipeline_state.x.pos[0][:2] - target)
        forward_reward = (dist_before - dist_after) / self.dt

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))

        obs = self._get_obs(pipeline_state, action)
        reward = forward_reward + healthy_reward - ctrl_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        success = jp.array(dist_after < 0.5, dtype=float)

        state.metrics.update(
            reward_forward=forward_reward,
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            x_position=pipeline_state.x.pos[0, 0],
            y_position=pipeline_state.x.pos[0, 1],
            dist_to_goal=dist_after,
            success=success,
        )
        state.info.update(info)
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: base.State, last_action: jax.Array) -> jax.Array:
        """State = [qpos (xyz+quat+8 joints), qvel (linvel+angvel+8 joints), last_action],
        followed by the goal channel [target_x, target_y]."""
        qpos = pipeline_state.q[:-2]
        qvel = pipeline_state.qd[:-2]
        target_pos = pipeline_state.x.pos[-1][:2]

        state = jp.concatenate([qpos, qvel, last_action])
        return jp.concatenate([state, target_pos])

    def _random_target(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Returns a target location in a random circle slightly above xy plane."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
        dist = 10
        ang = jp.pi * 2.0 * jax.random.uniform(rng2)
        target_x = dist * jp.cos(ang)
        target_y = dist * jp.sin(ang)
        return rng, jp.array([target_x, target_y])
