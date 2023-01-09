import itertools
import warnings
from typing import Dict, List

import pyspiel

import rm


def to_pyspiel_tab_policy(policy_list):
    return pyspiel.TabularPolicy(
        {
            istate: [
                (action, prob / max(1e-8, sum(as_and_ps.values())))
                for action, prob in as_and_ps.items()
            ]
            for istate, as_and_ps in itertools.chain(
            policy_list[0].items(), policy_list[1].items()
        )
        }
    )


def print_policy_profile(policy_profile: List[Dict[str, Dict[int, float]]], normalize: bool = True):
    for player, player_policy in enumerate(policy_profile):
        for infostate, policy in player_policy.items():
            prob_sum = 0.0
            for action, prob in policy.items():
                prob_sum += prob
            if prob_sum > 0:
                for action in policy.keys():
                    policy[action] /= prob_sum
        print("Player".upper(), player + 1)
        for infostate, dist in player_policy.items():
            print(
                rm.kuhn_poker_infostate_translation[(infostate, player)],
                "-->",
                list(
                    f"{rm.KuhnAction(action).name}: {round(prob, 2): .2f}"
                    for action, prob in dist.items()
                ),
            )


def print_final_policy_profile(policy_profile):
    alpha = policy_profile[0]["0"][1] / sum(policy_profile[0]["0"].values())
    if alpha > 1 / 3:
        warnings.warn(f"{alpha=} is greater than 1/3")
    else:
        print(f"{alpha=:.2f}")
    optimal_for_alpha = rm.kuhn_optimal_policy(alpha)
    print_policy_profile(policy_profile)
    print("\ntheoretically optimal policy:\n")
    for i, player_policy in optimal_for_alpha.items():
        print("Player".upper(), i + 1)
        for infostate, dist in player_policy.items():
            print(
                rm.kuhn_poker_infostate_translation[(infostate, i)],
                "-->",
                list(
                    f"{rm.KuhnAction(action).name}: {round(prob, 2): .2f}"
                    for action, prob in dist.items()
                ),
            )
    print("\nDifference to theoretically optimal policy:\n")
    for i, player_policy in enumerate(policy_profile):
        for infostate, avg_policy in player_policy.items():
            prob_sum = 0.0
            for action, prob in avg_policy.items():
                prob_sum += prob
            if prob_sum > 0:
                for action in avg_policy.keys():
                    avg_policy[action] /= prob_sum
        print("Player".upper(), i + 1)
        for infostate, dist in player_policy.items():
            print(
                rm.kuhn_poker_infostate_translation[(infostate, i)],
                "-->",
                list(
                    f"{rm.KuhnAction(action).name}: {round(prob - optimal_for_alpha[i][infostate][action], 2): .2f}"
                    for action, prob in dist.items()
                ),
            )
