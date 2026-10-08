import os
import pickle
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, NamedTuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import wandb_osh
from brax import envs
from brax.io import html
from etils import epath
from flax.training.train_state import TrainState
from wandb_osh.hooks import TriggerWandbSyncHook

import wandb
from buffer import TrajectoryUniformSamplingQueue
from envs.quantruped import Quantruped
from evaluator import CrlEvaluator
from leg_topology import (GOAL_DIM, LEG_NAMES, LOCAL_STATE_DIM, STATE_DIM,
                          assemble_action, local_state)
from networks import Actor, G_encoder, SA_encoder


@dataclass
class Args:
    exp_name: str = "train"
    seed: int = 1000
    track: bool = True
    wandb_project_name: str = "deep_rl_decentralized_crl"
    wandb_entity: str = ""
    wandb_mode: str = "online"
    wandb_dir: str = "."
    wandb_group: str = "."
    capture_vis: bool = True
    vis_length: int = 1000
    checkpoint: bool = True
    upload_checkpoints: bool = True  # final.pkl + args.pkl am Ende als wandb-Artifact hochladen
    upload_all_checkpoints: bool = True  # zusaetzlich alle Zwischen-Checkpoints (step_*.pkl) hochladen
    best_metric: str = "eval/episode_success"  # Eval-Metrik (maximiert), nach der best.pkl gewaehlt wird

    episode_length: int = 1000

    # Algorithm specific arguments
    total_env_steps: int = 100_000_000
    num_epochs: int = 100
    num_envs: int = 512
    num_eval_envs: int = 64
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    logsumexp_penalty_coeff: float = 0.1

    max_replay_size: int = 10000
    min_replay_size: int = 1000

    unroll_length: int = 62

    critic_network_width: int = 64
    actor_network_width: int = 64
    actor_depth: int = 8
    critic_depth: int = 8

    num_sgd_batches_per_training_step: int = 800

    entropy_param: float = 0.5
    disable_entropy: int = 0
    use_relu: int = 0
    num_render: int = 10
    save_buffer: int = 0

    # to be filled in runtime
    env_steps_per_actor_step: int = 0
    num_prefill_env_steps: int = 0
    num_prefill_actor_steps: int = 0
    num_training_steps_per_epoch: int = 0


@flax.struct.dataclass
class LegTrainState:
    """Independent Actor/Alpha params for a single decentralized leg
    controller -- no actor parameters are shared between legs."""

    actor_state: TrainState
    alpha_state: TrainState


@flax.struct.dataclass
class TrainingState:
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    legs: Dict[str, LegTrainState]
    critic_state: TrainState  # single global critic over full state + joint action


class Transition(NamedTuple):
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()


def checkpoint_params(training_state: TrainingState):
    """{leg: (alpha_params, actor_params), "critic": critic_params}"""
    params = {
        leg: (
            training_state.legs[leg].alpha_state.params,
            training_state.legs[leg].actor_state.params,
        )
        for leg in LEG_NAMES
    }
    params["critic"] = training_state.critic_state.params
    return params


def save_params(path: str, params: Any):
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


if __name__ == "__main__":
    args = tyro.cli(Args)

    print("Arguments:", flush=True)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}", flush=True)
    print("\n", flush=True)

    args.env_steps_per_actor_step = args.num_envs * args.unroll_length
    args.num_prefill_env_steps = args.min_replay_size * args.num_envs
    args.num_prefill_actor_steps = int(np.ceil(args.min_replay_size / args.unroll_length))
    args.num_training_steps_per_epoch = (
        args.total_env_steps - args.num_prefill_env_steps
    ) // (args.num_epochs * args.env_steps_per_actor_step)

    run_name = (
        f"quantruped_local_c{args.critic_depth}x{args.critic_network_width}"
        f"_a{args.actor_depth}x{args.actor_network_width}"
        f"_{args.batch_size}_{args.total_env_steps}_nenvs:{args.num_envs}_{args.seed}"
    )
    print(f"run_name: {run_name}", flush=True)

    if args.track:
        if args.wandb_group == ".":
            args.wandb_group = None
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity or None,
            mode=args.wandb_mode,
            group=args.wandb_group,
            dir=args.wandb_dir,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
        if args.wandb_mode == "offline":
            wandb_osh.set_log_level("ERROR")
            trigger_sync = TriggerWandbSyncHook()

    if args.checkpoint:
        from datetime import datetime
        from pathlib import Path

        short_run_name = f"runs/{run_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        save_path = Path(args.wandb_dir) / Path(short_run_name)
        os.makedirs(save_path, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, buffer_key, env_key, eval_env_key = jax.random.split(key, 4)
    key, critic_key, *leg_keys = jax.random.split(key, len(LEG_NAMES) + 2)

    def make_env():
        return Quantruped(backend="positional")

    env = envs.training.wrap(make_env(), episode_length=args.episode_length)
    obs_size = env.observation_size
    action_size = env.action_size
    assert obs_size == STATE_DIM + GOAL_DIM, (obs_size, STATE_DIM, GOAL_DIM)

    env_keys = jax.random.split(env_key, args.num_envs)
    env_state = jax.jit(env.reset)(env_keys)
    env.step = jax.jit(env.step)

    eval_env = envs.training.wrap(make_env(), episode_length=args.episode_length)
    eval_env.step = jax.jit(eval_env.step)

    # Network + TrainState setup: one independent Actor/Alpha per leg, one global Critic.
    actor = Actor(action_size=2, network_width=args.actor_network_width, network_depth=args.actor_depth, use_relu=args.use_relu)
    sa_encoder = SA_encoder(network_width=args.critic_network_width, network_depth=args.critic_depth, use_relu=args.use_relu)
    g_encoder = G_encoder(network_width=args.critic_network_width, network_depth=args.critic_depth, use_relu=args.use_relu)

    target_entropy = -args.entropy_param * 2  # 2 actuators per leg

    leg_states = {}
    for leg, actor_key in zip(LEG_NAMES, leg_keys):
        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(actor_key, np.ones([1, LOCAL_STATE_DIM + GOAL_DIM])),
            tx=optax.adam(learning_rate=args.actor_lr),
        )
        log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
        alpha_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": log_alpha},
            tx=optax.adam(learning_rate=args.alpha_lr),
        )
        leg_states[leg] = LegTrainState(actor_state=actor_state, alpha_state=alpha_state)

    sa_key, g_key = jax.random.split(critic_key)
    sa_params = sa_encoder.init(sa_key, np.ones([1, STATE_DIM]), np.ones([1, action_size]))
    g_params = g_encoder.init(g_key, np.ones([1, GOAL_DIM]))
    critic_state = TrainState.create(
        apply_fn=None,
        params={"sa_encoder": sa_params, "g_encoder": g_params},
        tx=optax.adam(learning_rate=args.critic_lr),
    )

    training_state = TrainingState(
        env_steps=jnp.zeros(()),
        gradient_steps=jnp.zeros(()),
        legs=leg_states,
        critic_state=critic_state,
    )

    # Replay Buffer (shared across legs: stores full observation/action vectors).
    dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))
    dummy_transition = Transition(
        observation=dummy_obs,
        action=dummy_action,
        reward=0.0,
        discount=0.0,
        extras={"state_extras": {"truncation": 0.0, "seed": 0.0}},
    )

    def jit_wrap(buffer):
        buffer.insert_internal = jax.jit(buffer.insert_internal)
        buffer.sample_internal = jax.jit(buffer.sample_internal)
        return buffer

    replay_buffer = jit_wrap(
        TrajectoryUniformSamplingQueue(
            max_replay_size=args.max_replay_size,
            dummy_data_sample=dummy_transition,
            sample_batch_size=args.batch_size,
            num_envs=args.num_envs,
            episode_length=args.episode_length,
        )
    )
    buffer_state = jax.jit(replay_buffer.init)(buffer_key)

    def split_obs(obs):
        return obs[..., :STATE_DIM], obs[..., STATE_DIM:STATE_DIM + GOAL_DIM]

    def deterministic_actor_step(training_state, env, env_state, extra_fields):
        state, goal = split_obs(env_state.obs)
        actions_per_leg = {}
        for leg in LEG_NAMES:
            actor_input = jnp.concatenate([local_state(state, leg), goal], axis=-1)
            means, _ = actor.apply(training_state.legs[leg].actor_state.params, actor_input)
            actions_per_leg[leg] = nn.tanh(means)
        actions = assemble_action(actions_per_leg)

        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        return nstate, Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            extras={"state_extras": state_extras},
        )

    def actor_step(training_state, env, env_state, key, extra_fields):
        state, goal = split_obs(env_state.obs)
        actions_per_leg = {}
        keys = jax.random.split(key, len(LEG_NAMES))
        for leg, leg_key in zip(LEG_NAMES, keys):
            actor_input = jnp.concatenate([local_state(state, leg), goal], axis=-1)
            means, log_stds = actor.apply(training_state.legs[leg].actor_state.params, actor_input)
            stds = jnp.exp(log_stds)
            actions_per_leg[leg] = nn.tanh(
                means + stds * jax.random.normal(leg_key, shape=means.shape, dtype=means.dtype)
            )
        actions = assemble_action(actions_per_leg)

        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        return nstate, Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            extras={"state_extras": state_extras},
        )

    @jax.jit
    def get_experience(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused_t):
            env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            env_state, transition = actor_step(
                training_state, env, env_state, current_key, extra_fields=("truncation", "seed")
            )
            return (env_state, next_key), transition

        (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=args.unroll_length)
        buffer_state = replay_buffer.insert(buffer_state, data)
        return env_state, buffer_state

    def prefill_replay_buffer(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused):
            del unused
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            env_state, buffer_state = get_experience(training_state, env_state, buffer_state, key)
            training_state = training_state.replace(
                env_steps=training_state.env_steps + args.env_steps_per_actor_step,
            )
            return (training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(f, (training_state, env_state, buffer_state, key), (), length=args.num_prefill_actor_steps)[0]

    @jax.jit
    def update_actor_and_alpha(transitions, training_state, key):
        state = transitions.observation[:, :STATE_DIM]
        future_state = transitions.extras["future_state"]
        goal = future_state[:, 0:GOAL_DIM]
        critic_params = training_state.critic_state.params

        # Joint loss over all leg actors: the joint action is sampled from the
        # current per-leg policies and scored by the global critic, so each
        # actor receives the critic gradient through its own action slice.
        def actor_loss(actor_params, log_alphas, key):
            keys = jax.random.split(key, len(LEG_NAMES))
            actions_per_leg = {}
            log_probs = {}
            for leg, leg_key in zip(LEG_NAMES, keys):
                actor_input = jnp.concatenate([local_state(state, leg), goal], axis=1)
                means, log_stds = actor.apply(actor_params[leg], actor_input)
                stds = jnp.exp(log_stds)
                x_ts = means + stds * jax.random.normal(leg_key, shape=means.shape, dtype=means.dtype)
                leg_action = nn.tanh(x_ts)
                log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
                log_prob -= jnp.log((1 - jnp.square(leg_action)) + 1e-6)
                actions_per_leg[leg] = leg_action
                log_probs[leg] = log_prob.sum(-1)
            action = assemble_action(actions_per_leg)

            sa_repr = sa_encoder.apply(critic_params["sa_encoder"], state, action)
            g_repr = g_encoder.apply(critic_params["g_encoder"], goal)
            qf_pi = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))

            if args.disable_entropy:
                loss = -jnp.mean(qf_pi)
            else:
                entropy_term = sum(jnp.exp(log_alphas[leg]) * log_probs[leg] for leg in LEG_NAMES)
                loss = jnp.mean(entropy_term - qf_pi)
            return loss, log_probs

        def alpha_loss(alpha_params, log_prob):
            alpha = jnp.exp(alpha_params["log_alpha"])
            return jnp.mean(alpha * jnp.mean(jax.lax.stop_gradient(-log_prob - target_entropy)))

        actor_params = {leg: training_state.legs[leg].actor_state.params for leg in LEG_NAMES}
        log_alphas = {leg: training_state.legs[leg].alpha_state.params["log_alpha"] for leg in LEG_NAMES}
        (actorloss, log_probs), actor_grads = jax.value_and_grad(actor_loss, has_aux=True)(
            actor_params, log_alphas, key
        )

        new_legs = {}
        metrics = {"actor_loss": actorloss}
        for leg in LEG_NAMES:
            leg_state = training_state.legs[leg]
            new_actor_state = leg_state.actor_state.apply_gradients(grads=actor_grads[leg])

            alphaloss, alpha_grad = jax.value_and_grad(alpha_loss)(leg_state.alpha_state.params, log_probs[leg])
            new_alpha_state = leg_state.alpha_state.apply_gradients(grads=alpha_grad)

            new_legs[leg] = leg_state.replace(actor_state=new_actor_state, alpha_state=new_alpha_state)
            metrics[f"{leg}/alpha_loss"] = alphaloss
            metrics[f"{leg}/sample_entropy"] = -log_probs[leg]

        training_state = training_state.replace(legs=new_legs)
        return training_state, metrics

    @jax.jit
    def update_critic(transitions, training_state, key):
        state = transitions.observation[:, :STATE_DIM]
        goal = transitions.observation[:, STATE_DIM:STATE_DIM + GOAL_DIM]
        action = transitions.action

        def critic_loss(critic_params):
            sa_repr = sa_encoder.apply(critic_params["sa_encoder"], state, action)
            g_repr = g_encoder.apply(critic_params["g_encoder"], goal)

            logits = -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1))
            loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))

            logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
            loss += args.logsumexp_penalty_coeff * jnp.mean(logsumexp ** 2)
            return loss, logsumexp

        (loss, logsumexp), grad = jax.value_and_grad(critic_loss, has_aux=True)(training_state.critic_state.params)
        new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
        training_state = training_state.replace(critic_state=new_critic_state)
        metrics = {"critic_loss": loss, "logsumexp": logsumexp.mean()}
        return training_state, metrics

    @jax.jit
    def sgd_step(carry, transitions):
        training_state, key = carry
        key, critic_key, actor_key = jax.random.split(key, 3)

        training_state, actor_metrics = update_actor_and_alpha(transitions, training_state, actor_key)
        training_state, critic_metrics = update_critic(transitions, training_state, critic_key)
        training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)

        metrics = {}
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)
        return (training_state, key), metrics

    @jax.jit
    def training_step(training_state, env_state, buffer_state, key):
        experience_key1, experience_key2, sampling_key, training_key, sgd_batches_key = jax.random.split(key, 5)

        env_state, buffer_state = get_experience(training_state, env_state, buffer_state, experience_key1)
        training_state = training_state.replace(env_steps=training_state.env_steps + args.env_steps_per_actor_step)

        buffer_state, transitions = replay_buffer.sample(buffer_state)

        batch_keys = jax.random.split(sampling_key, transitions.observation.shape[0])
        transitions = jax.vmap(TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, 0, 0))(
            (args.gamma, STATE_DIM, 0, GOAL_DIM), transitions, batch_keys
        )
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions
        )

        permutation = jax.random.permutation(experience_key2, len(transitions.observation))
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)

        num_full_batches = len(transitions.observation) // args.batch_size
        transitions = jax.tree_util.tree_map(lambda x: x[: num_full_batches * args.batch_size], transitions)
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, args.batch_size) + x.shape[1:]), transitions
        )

        num_total_batches = transitions.observation.shape[0]
        selected_indices = jax.random.permutation(sgd_batches_key, num_total_batches)[
            : args.num_sgd_batches_per_training_step
        ]
        transitions = jax.tree_util.tree_map(lambda x: x[selected_indices], transitions)

        (training_state, _), metrics = jax.lax.scan(sgd_step, (training_state, training_key), transitions)
        return (training_state, env_state, buffer_state), metrics

    @jax.jit
    def training_epoch(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused_t):
            ts, es, bs, k = carry
            k, train_key = jax.random.split(k, 2)
            (ts, es, bs), metrics = training_step(ts, es, bs, train_key)
            return (ts, es, bs, k), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
            f, (training_state, env_state, buffer_state, key), jnp.arange(args.num_training_steps_per_epoch)
        )
        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        return training_state, env_state, buffer_state, metrics

    key, prefill_key = jax.random.split(key, 2)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_key
    )

    evaluator = CrlEvaluator(
        deterministic_actor_step,
        eval_env,
        num_eval_envs=args.num_eval_envs,
        episode_length=args.episode_length,
        key=eval_env_key,
    )

    training_walltime = 0
    best_value, best_epoch, best_env_steps = -np.inf, None, None
    print("starting training....", flush=True)
    start_time = time.time()
    for ne in range(args.num_epochs):
        t = time.time()
        key, epoch_key = jax.random.split(key)
        training_state, env_state, buffer_state, metrics = training_epoch(training_state, env_state, buffer_state, epoch_key)

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (args.env_steps_per_actor_step * args.num_training_steps_per_epoch) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            "training/envsteps": training_state.env_steps.item(),
            **{f"training/{name}": value for name, value in metrics.items()},
        }

        metrics = evaluator.run_evaluation(training_state, metrics)
        print(f"epoch {ne} out of {args.num_epochs} complete. metrics: {metrics}", flush=True)

        if args.checkpoint:
            if ne < 5 or ne >= args.num_epochs - 5 or ne % 10 == 0:
                params = checkpoint_params(training_state)
                path = f"{save_path}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)

            # Best checkpoint: ueberschreiben, sobald die Eval-Metrik mindestens so gut ist wie bisher.
            value = float(metrics.get(args.best_metric, np.nan))
            if not np.isnan(value) and value >= best_value:  # bei Gleichstand gewinnt der spaetere
                best_value, best_epoch, best_env_steps = value, ne, int(training_state.env_steps)
                save_params(f"{save_path}/best.pkl", checkpoint_params(training_state))
                print(f"New best {args.best_metric}={value:.4f} in epoch {ne}, saved best.pkl", flush=True)
                if args.track:
                    wandb.run.summary["best/value"] = best_value
                    wandb.run.summary["best/epoch"] = best_epoch
                    wandb.run.summary["best/env_steps"] = best_env_steps

        if args.track:
            wandb.log(metrics, step=ne)
            if args.wandb_mode == "offline":
                trigger_sync()

        hours_passed = (time.time() - start_time) / 3600
        print(f"Time elapsed: {hours_passed:.3f} hours", flush=True)

    if args.checkpoint:
        params = checkpoint_params(training_state)
        save_params(f"{save_path}/final.pkl", params)

    if args.capture_vis:
        def render_policy(training_state, save_path):
            render_env = make_env()
            render_env = envs.training.EpisodeWrapper(
                render_env, args.episode_length, action_repeat=1
            )
            render_env = envs.training.AutoResetWrapper(render_env)

            @jax.jit
            def policy_step(env_state, training_state):
                state, goal = split_obs(env_state.obs)
                actions_per_leg = {}
                for leg in LEG_NAMES:
                    actor_input = jnp.concatenate([local_state(state, leg), goal], axis=-1)
                    means, _ = actor.apply(training_state.legs[leg].actor_state.params, actor_input)
                    actions_per_leg[leg] = nn.tanh(means)
                actions = assemble_action(actions_per_leg)
                next_state = render_env.step(env_state, actions)
                return next_state, env_state

            rollout_states = []
            for i in range(args.num_render):
                rng = jax.random.PRNGKey(seed=i + 1)
                render_env_state = jax.jit(render_env.reset)(rng)
                for _ in range(args.vis_length):
                    render_env_state, current_state = policy_step(render_env_state, training_state)
                    rollout_states.append(current_state.pipeline_state)

            html_string = html.render(render_env.sys, rollout_states)
            with open(f"{save_path}/vis.html", "w") as f:
                f.write(html_string)
            if args.track:
                wandb.log({"vis": wandb.Html(html_string)})

        print("Rendering final policy...", flush=True)
        try:
            render_policy(training_state, save_path)
        except Exception as e:
            print(f"Error rendering final policy: {e}", flush=True)

    if args.checkpoint:
        with open(f"{save_path}/args.pkl", "wb") as f:
            pickle.dump(args, f)
        print(f"Saved args to {save_path}/args.pkl", flush=True)

    if args.checkpoint and args.track and args.upload_checkpoints:
        # Checkpoints als wandb-Artifact hochladen, damit man sie ohne laufende
        # Instanz herunterladen kann (wandb.ai -> Run -> Artifacts).
        try:
            artifact = wandb.Artifact(
                name=f"checkpoints-{wandb.run.id}",
                type="model",
                metadata={
                    "env_steps": int(training_state.env_steps),
                    "seed": args.seed,
                    "num_envs": args.num_envs,
                    "actor_depth": args.actor_depth,
                    "critic_depth": args.critic_depth,
                    "actor_network_width": args.actor_network_width,
                    "critic_network_width": args.critic_network_width,
                    "best_metric": args.best_metric,
                    "best_value": None if best_epoch is None else best_value,
                    "best_epoch": best_epoch,
                    "best_env_steps": best_env_steps,
                },
            )
            artifact.add_file(f"{save_path}/final.pkl")
            if best_epoch is not None:
                artifact.add_file(f"{save_path}/best.pkl")
            artifact.add_file(f"{save_path}/args.pkl")
            if args.upload_all_checkpoints:
                for ckpt in sorted(Path(save_path).glob("step_*.pkl")):
                    artifact.add_file(str(ckpt))
            wandb.log_artifact(artifact, aliases=["latest", "final"])
            print("Uploading checkpoints to wandb ...", flush=True)
            wandb.finish()  # wartet, bis der Upload abgeschlossen ist
            print("Checkpoints uploaded to wandb.", flush=True)
        except Exception as e:
            print(f"Error uploading checkpoints to wandb: {e}", flush=True)
