"""Standalone evaluation/rendering for a saved Quantruped checkpoint.

Loads a checkpoint (e.g. runs/<run>/final.pkl or step_*.pkl) together with
the matching args.pkl from the same run, runs the CrlEvaluator to report
metrics, and optionally renders rollouts to an HTML visualization.
"""
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import tyro
from brax import envs
from brax.io import html

from envs.quantruped import Quantruped
from evaluator import CrlEvaluator
from leg_topology import (GOAL_DIM, LEG_NAMES, STATE_DIM, assemble_action,
                          local_state)
from networks import Actor
from train import Args, Transition


@dataclass
class EvalArgs:
    checkpoint: str
    """Path to a saved checkpoint .pkl, e.g. runs/<run>/final.pkl."""
    args_path: Optional[str] = None
    """Path to the matching args.pkl. Defaults to args.pkl next to the checkpoint."""
    seed: int = 0
    num_eval_envs: int = 128
    episode_length: Optional[int] = None
    """Overrides the training episode length if set."""
    render: bool = True
    num_render: int = 10
    vis_length: int = 1000
    out_dir: str = "."


def make_env(backend: str = "positional") -> Quantruped:
    return Quantruped(backend=backend)


def split_obs(obs):
    return obs[..., :STATE_DIM], obs[..., STATE_DIM:STATE_DIM + GOAL_DIM]


def load_train_args(path: str) -> Args:
    # train.py pickles Args while running as __main__, so unpickling it
    # elsewhere needs __main__.Args to resolve to the same class.
    import __main__
    __main__.Args = Args
    with open(path, "rb") as f:
        return pickle.load(f)


def load_params(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_actions_fn(actor: Actor, params):
    def actions_fn(obs):
        state, goal = split_obs(obs)
        actions_per_leg = {}
        for leg in LEG_NAMES:
            actor_input = jnp.concatenate([local_state(state, leg), goal], axis=-1)
            means, _ = actor.apply(params[leg][1], actor_input)
            actions_per_leg[leg] = nn.tanh(means)
        return assemble_action(actions_per_leg)

    return actions_fn


def main():
    eval_args = tyro.cli(EvalArgs)

    args_path = eval_args.args_path or os.path.join(
        os.path.dirname(eval_args.checkpoint), "args.pkl"
    )
    train_args = load_train_args(args_path)
    params = load_params(eval_args.checkpoint)
    episode_length = eval_args.episode_length or train_args.episode_length

    actor = Actor(
        action_size=2,
        network_width=train_args.actor_network_width,
        network_depth=train_args.actor_depth,
        use_relu=train_args.use_relu,
    )
    actions_fn = build_actions_fn(actor, params)

    def deterministic_actor_step(_training_state, env, env_state, extra_fields):
        actions = actions_fn(env_state.obs)
        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        return nstate, Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            extras={"state_extras": state_extras},
        )

    eval_env = envs.training.wrap(make_env(), episode_length=episode_length)
    eval_env.step = jax.jit(eval_env.step)

    key = jax.random.PRNGKey(eval_args.seed)
    evaluator = CrlEvaluator(
        deterministic_actor_step,
        eval_env,
        num_eval_envs=eval_args.num_eval_envs,
        episode_length=episode_length,
        key=key,
    )

    metrics = evaluator.run_evaluation(training_state=None, training_metrics={})
    print("\nEvaluation metrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v}")

    if eval_args.render:
        # EpisodeWrapper + AutoResetWrapper (no VmapWrapper, single unbatched
        # env) so the rollout resets instead of continuing physics past a
        # fall, which otherwise diverges to NaN within a few dozen steps.
        render_env = make_env()
        render_env = envs.training.EpisodeWrapper(render_env, episode_length, action_repeat=1)
        render_env = envs.training.AutoResetWrapper(render_env)

        @jax.jit
        def policy_step(env_state):
            actions = actions_fn(env_state.obs)
            next_state = render_env.step(env_state, actions)
            return next_state, env_state

        rollout_states = []
        for i in range(eval_args.num_render):
            rng = jax.random.PRNGKey(eval_args.seed + i + 1)
            render_env_state = jax.jit(render_env.reset)(rng)
            for _ in range(eval_args.vis_length):
                render_env_state, current_state = policy_step(render_env_state)
                rollout_states.append(current_state.pipeline_state)

        html_string = html.render(render_env.sys, rollout_states)
        out_path = os.path.join(eval_args.out_dir, "eval_vis.html")
        with open(out_path, "w") as f:
            f.write(html_string)
        print(f"\nSaved visualization to {out_path}")


if __name__ == "__main__":
    main()
