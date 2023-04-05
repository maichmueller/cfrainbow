import math
from typing import (
    Optional,
    MutableMapping,
)

from ..spiel_types import *


_POS_THRESH = 1e-8


def normalize_or_uniformize_policy(policy, pos_regret_dict, pos_regret_sum):
    if pos_regret_sum > _POS_THRESH:
        for action, pos_regret in pos_regret_dict.items():
            policy[action] = pos_regret / pos_regret_sum
    else:
        uniform_prob = 1 / len(policy)
        for action in policy.keys():
            policy[action] = uniform_prob


def regret_matching(
    policy: MutableMapping[Action, Probability],
    cumul_regret_map: Mapping[Action, float],
    positivized_regret_map: Optional[MutableMapping[Action, float]] = None,
):
    pos_regret_dict = (
        positivized_regret_map if positivized_regret_map is not None else dict()
    )
    pos_regret_sum = 0.0
    for action, regret in cumul_regret_map.items():
        pos_action_regret = max(0.0, regret)
        pos_regret_dict[action] = pos_action_regret
        pos_regret_sum += pos_action_regret
    normalize_or_uniformize_policy(policy, pos_regret_dict, pos_regret_sum)


def regret_matching_plus(
    policy: MutableMapping[Action, Probability],
    cumul_regret_map: MutableMapping[Action, float],
):
    regret_matching(policy, cumul_regret_map, cumul_regret_map)


def predictive_regret_matching(
    prediction: Mapping[Action, float],
    policy: MutableMapping[Action, Probability],
    cumul_regret_map: Mapping[Action, float],
):
    pos_regret_sum = 0.0
    expected_prediction: float = sum(
        [prediction[action] * action_prob for action, action_prob in policy.items()]
    )
    positivized_regret_map = dict()
    for action, regret in cumul_regret_map.items():
        pos_regret = max(0.0, regret + (prediction[action] - expected_prediction))
        pos_regret_sum += pos_regret
        positivized_regret_map[action] = pos_regret
    normalize_or_uniformize_policy(policy, positivized_regret_map, pos_regret_sum)


def predictive_regret_matching_plus(
    prediction: Mapping[Action, float],
    policy: MutableMapping[Action, Probability],
    cumul_regret_map: MutableMapping[Action, float],
):
    pos_regret_sum = 0.0
    avg_prediction = sum(
        [prediction[action] * action_prob for action, action_prob in policy.items()]
    )
    positivized_regret_map = dict()
    for action, regret in cumul_regret_map.items():
        pos_regret = max(0.0, regret)
        cumul_regret_map[action] = pos_regret
        positivized_regret_map[action] = pred_pos_regret = max(
            0.0, pos_regret + (prediction[action] - avg_prediction)
        )
        pos_regret_sum += pred_pos_regret
    normalize_or_uniformize_policy(policy, positivized_regret_map, pos_regret_sum)


def hedge(
    policy: MutableMapping[Action, Probability],
    cumul_utility_map: Mapping[Action, float],
    lr: float,
):
    sum_weights = 0.0
    for action, utility in cumul_utility_map.items():
        new_weight = math.exp(-lr * utility)
        sum_weights += new_weight
        policy[action] = new_weight
    for action in policy:
        policy[action] /= sum_weights
