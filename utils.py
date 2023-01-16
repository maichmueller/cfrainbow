import itertools
import warnings
from enum import Enum
from functools import reduce, singledispatchmethod
from typing import Dict, List, Union, Any, Sequence

import numpy as np
import pyspiel
from numba import njit

import rm


class KuhnAction(Enum):
    check = 0
    bet = 1


class LeducAction(Enum):
    Fold = 0
    Call = 1
    Raise = 2


def counterfactual_reach_prob(reach_prob_map: dict[int, float], player: int):
    return reduce(
        lambda x, y: x * y,
        [rp for i, rp in reach_prob_map.items() if i != player],
        1.0,
    )


def kuhn_optimal_policy(alpha: float):
    policy = [dict(), dict()]
    # player 0
    policy[0] = {
        "0": {0: 1.0 - alpha, 1: alpha},
        "0pb": {0: 1.0, 1: 0.0},
        "1": {0: 1.0, 1: 0.0},
        "1pb": {0: 2.0 / 3.0 - alpha, 1: 1.0 / 3.0 + alpha},
        "2": {0: 1.0 - 3.0 * alpha, 1: 3.0 * alpha},
        "2pb": {0: 0.0, 1: 1.0},
    }

    # player 1
    policy[1] = {
        "0p": {0: 2.0 / 3.0, 1: 1.0 / 3.0},
        "0b": {0: 1, 1: 0},
        "1p": {0: 1, 1: 0},
        "1b": {0: 2.0 / 3.0, 1: 1.0 / 3.0},
        "2p": {0: 0, 1: 1},
        "2b": {0: 0, 1: 1},
    }
    return policy


def all_states_gen(
    root=None,
    include_chance_states: bool = False,
    include_terminal_states: bool = False,
    game: Union[pyspiel.Game, str] = "kuhn_poker",
):
    if root is None:
        root = (
            pyspiel.load_game(game) if isinstance(game, str) else game
        ).new_initial_state()
    stack = [root]
    while stack:
        s = stack.pop()
        if s.is_terminal():
            if include_terminal_states:
                yield s
            continue
        if s.is_chance_node():
            if include_chance_states:
                yield s
            for outcome, prob in s.chance_outcomes():
                child = s.child(outcome)
                stack.append(child)
        else:
            yield s
            for action in s.legal_actions():
                child = s.child(action)
                stack.append(child)


class KuhnTensorToStr:
    def __init__(self):
        self.tensors = []
        self.strs = []
        for state in all_states_gen():
            self.tensors.append(
                tuple(state.information_state_tensor(state.current_player()))
            )
            self.strs.append(state.information_state_string(state.current_player()))

    @singledispatchmethod
    def __getitem__(self, istate):
        raise NotImplementedError

    @__getitem__.register(str)
    def _(self, infostate_str):
        where = self.strs.index(infostate_str)
        return self.tensors[where]

    @__getitem__.register(tuple)
    def _(self, infostate_str):
        where = self.tensors.index(infostate_str)
        return self.strs[where]


class KuhnLeducHistoryToStr:
    def __init__(self, game_name: str = "kuhn_poker"):
        self.tensors = []
        self.strs = []
        for state in all_states_gen(game=game_name):
            self.tensors.append(tuple(self.istate_as_action_observation(state)))
            self.strs.append(state.information_state_string(state.current_player()))

    @staticmethod
    def istate_as_action_observation(state: pyspiel.State):
        history = list((entry.action, entry.player) for entry in state.full_history())
        # in kuhn poker...
        # the first action is the chance player assigning a card to player 0
        # the second action is the chance player assigning a card to player 1
        # depending on the active player, we have to hide the info given here of the other player's card
        # we mask an unknown action as simply -1
        if state.current_player() == 0:
            entry_to_hide = history[1]
            history[1] = -1, entry_to_hide[1]
        else:
            # active player is player 1
            entry_to_hide = history[0]
            history[0] = -1, entry_to_hide[1]
        return history

    @singledispatchmethod
    def __getitem__(self, istate):
        raise NotImplementedError

    @__getitem__.register(str)
    def _(self, infostate_str):
        where = self.strs.index(infostate_str)
        return self.tensors[where]

    @__getitem__.register(tuple)
    def _(self, infostate_str):
        where = self.tensors.index(infostate_str)
        return self.strs[where]


kuhn_tensor_to_str_bij = KuhnTensorToStr()
kuhn_history_to_str_bij = KuhnLeducHistoryToStr()
leduc_history_to_str_bij = KuhnLeducHistoryToStr(game_name="leduc_poker")

fill_len = len(f"P1: {'?'.center(5, ' ')} | P2: {'?'.center(5, ' ')} | c b ")
kuhn_poker_infostate_translation = {
    k: v.ljust(fill_len, " ")
    for k, v in {
        ("0", 0): f"P1: Jack  | P2: {'?'.center(5, ' ')}",
        ("1", 0): f"P1: Queen | P2: {'?'.center(5, ' ')}",
        ("2", 0): f"P1: King  | P2: {'?'.center(5, ' ')}",
        ("0pb", 0): f"P1: Jack  | P2: {'?'.center(5, ' ')} | cb",
        ("1pb", 0): f"P1: Queen | P2: {'?'.center(5, ' ')} | cb",
        ("2pb", 0): f"P1: King  | P2: {'?'.center(5, ' ')} | cb",
        ("0p", 1): f"P1: {'?'.center(5, ' ')} | P2: Jack  | c",
        ("1p", 1): f"P1: {'?'.center(5, ' ')} | P2: Queen | c",
        ("2p", 1): f"P1: {'?'.center(5, ' ')} | P2: King  | c",
        ("0b", 1): f"P1: {'?'.center(5, ' ')} | P2: Jack  | b",
        ("1b", 1): f"P1: {'?'.center(5, ' ')} | P2: Queen | b",
        ("2b", 1): f"P1: {'?'.center(5, ' ')} | P2: King  | b",
    }.items()
}


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


def sample_on_policy(values: Sequence[Any], policy: Sequence[float], rng: np.random.Generator, epsilon: float = 0.0):
    if epsilon != 0.0:
        uniform_prob = 1.0 / len(policy)
        policy = [epsilon * uniform_prob + (1 - epsilon) * policy_prob for policy_prob in policy]
    choice = rng.choice(np.arange(len(values)), p=policy)
    return values[choice], choice, policy


def print_kuhn_poker_policy_profile(policy_profile: List[Dict[str, Dict[int, float]]]):
    for player, player_policy in enumerate(policy_profile):
        print("Player".upper(), player + 1)
        for infostate, action_policy in player_policy.items():
            print(
                kuhn_poker_infostate_translation[(infostate, player)],
                "-->",
                list(
                    f"{KuhnAction(action).name}: {round(prob, 2): .2f}"
                    for action, prob in action_policy.items()
                ),
            )


def normalize_action_policy(action_policy):
    norm_action_policy = {}
    prob_sum = sum(action_policy.values())
    if prob_sum <= 0:
        raise ValueError(f"Policy values summed up to {prob_sum} <= 0.")
    for action, prob in action_policy.items():
        norm_action_policy[action] = prob / prob_sum
    return norm_action_policy


def normalize_state_policy(state_policy):
    norm_policy = {}
    for infostate, action_policy in state_policy.items():
        norm_action_policy = normalize_action_policy(action_policy)
        norm_policy[infostate] = norm_action_policy
    return norm_policy


def normalize_policy_profile(policy_profile):
    norm_policy_profile = []
    for player, state_policy in enumerate(policy_profile):
        norm_action_policy = normalize_state_policy(state_policy)
        norm_policy_profile.append(norm_action_policy)
    return norm_policy_profile


def print_final_policy_profile(policy_profile):
    alpha = policy_profile[0]["0"][1] / sum(policy_profile[0]["0"].values())
    if alpha > 1 / 3:
        warnings.warn(f"{alpha=} is greater than 1/3")
    else:
        print(f"{alpha=:.2f}")
    optimal_for_alpha = kuhn_optimal_policy(alpha)
    normalized_policy_profile = normalize_policy_profile(policy_profile)
    print_kuhn_poker_policy_profile(normalized_policy_profile)
    print("\ntheoretically optimal policy:\n")
    print_kuhn_poker_policy_profile(optimal_for_alpha)
    print("\nDifference to theoretically optimal policy:\n")
    for i, player_policy in enumerate(normalized_policy_profile):
        print("Player".upper(), i + 1)
        for infostate, dist in player_policy.items():
            print(
                kuhn_poker_infostate_translation[(infostate, i)],
                "-->",
                list(
                    f"{KuhnAction(action).name}: {round(prob - optimal_for_alpha[i][infostate][action], 2): .2f}"
                    for action, prob in dist.items()
                ),
            )
