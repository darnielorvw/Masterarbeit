"""Decentralization topology for the four leg controllers, mirroring the
observation/action scope of `Quantruped_Local_Env` from the original ddrl
repo (deep_rl/env/quantruped_fourDecentralizedController_environments.py):
each leg controller observes its own leg plus both ring-neighboring legs,
in addition to global torso information.

Indices below refer to the 37-dim *state* part of the Quantruped observation
(see envs/quantruped.py:_get_obs), laid out as:
  [0:3]    torso xyz
  [3:7]    torso quaternion
  [7:9]    FL (hip, knee) qpos      [9:11]  HL qpos   [11:13] HR qpos   [13:15] FR qpos
  [15:18]  torso linear velocity
  [18:21]  torso angular velocity
  [21:23]  FL qvel   [23:25] HL qvel   [25:27] HR qvel   [27:29] FR qvel
  [29:31]  FR last_action   [31:33] FL last_action   [33:35] HL last_action   [35:37] HR last_action
           (actuator order, see ACTION_ORDER -- differs from the qpos/qvel joint order)
The goal channel (target x, y) is appended after these 37 dims and is handled
separately (shared across all legs).
"""

import numpy as np

LEG_NAMES = ["FL", "HL", "HR", "FR"]

# Ring adjacency FL <-> HL <-> HR <-> FR <-> FL: each leg's two neighbors.
NEIGHBORS = {
    "FL": ("HL", "FR"),
    "HL": ("HR", "FL"),
    "HR": ("FR", "HL"),
    "FR": ("FL", "HR"),
}

GLOBAL_IDX = [0, 1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19, 20]

LEG_QPOS_IDX = {"FL": (7, 8), "HL": (9, 10), "HR": (11, 12), "FR": (13, 14)}
LEG_QVEL_IDX = {"FL": (21, 22), "HL": (23, 24), "HR": (25, 26), "FR": (27, 28)}
# Action order in the XML's <actuator> block (see envs/assets/quantruped.xml):
# [hip_4/ankle_4, hip_1/ankle_1, hip_2/ankle_2, hip_3/ankle_3] = [FR, FL, HL, HR].
ACTION_ORDER = ["FR", "FL", "HL", "HR"]

# last_action is stored in actuator order, unlike qpos/qvel (joint order).
LEG_ACTION_IDX = {leg: (29 + 2 * i, 30 + 2 * i) for i, leg in enumerate(ACTION_ORDER)}

STATE_DIM = 37
GOAL_DIM = 2


def _leg_block(leg: str):
    return [*LEG_QPOS_IDX[leg], *LEG_QVEL_IDX[leg], *LEG_ACTION_IDX[leg]]


def _build_local_indices():
    indices = {}
    for leg in LEG_NAMES:
        idx = list(GLOBAL_IDX) + _leg_block(leg)
        for neighbor in NEIGHBORS[leg]:
            idx += _leg_block(neighbor)
        indices[leg] = np.array(idx, dtype=np.int32)
    return indices


# Per-leg indices into the 37-dim state vector: global torso info + own leg +
# both ring neighbors (13 + 3 * 6 = 31 dims).
LOCAL_STATE_INDICES = _build_local_indices()
LOCAL_STATE_DIM = LOCAL_STATE_INDICES[LEG_NAMES[0]].shape[0]

# Index of each leg's (hip, knee) pair within the assembled 8-dim action vector
# (distinct from LEG_ACTION_IDX, which indexes the last-action *feature* inside
# the 37-dim state vector).
ACTION_VECTOR_IDX = {
    leg: (2 * i, 2 * i + 1) for i, leg in enumerate(ACTION_ORDER)
}


def assemble_action(actions_per_leg):
    """actions_per_leg: dict[leg_name] -> array(..., 2). Returns array(..., 8)
    in the XML actuator order, matching the original `concatenate_actions`."""
    import jax.numpy as jp

    return jp.concatenate([actions_per_leg[leg] for leg in ACTION_ORDER], axis=-1)


def local_state(state: "jax.Array", leg: str):
    """Slice the 37-dim state part of an observation down to `leg`'s local view."""
    return state[..., LOCAL_STATE_INDICES[leg]]


def local_action(action: "jax.Array", leg: str):
    """Slice `leg`'s (hip, knee) pair out of the assembled 8-dim action vector."""
    idx = ACTION_VECTOR_IDX[leg]
    return action[..., idx[0]:idx[1] + 1]
