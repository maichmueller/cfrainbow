from enum import Enum
from functools import reduce, singledispatchmethod
from typing import Union

import pyspiel


class KuhnAction(Enum):
    check = 0
    bet = 1


class LeducAction(Enum):
    Fold = 0
    Call = 1
    Raise = 2


def regret_matching(policy, regret_dict):
    pos_regret_dict = dict()
    pos_regret_sum = 0.0
    for action, regret in regret_dict.items():
        pos_action_regret = max(0, regret)
        pos_regret_dict[action] = pos_action_regret
        pos_regret_sum += pos_action_regret
    if pos_regret_sum > 0.0:
        for action, pos_regret in pos_regret_dict.items():
            policy[action] = pos_regret / pos_regret_sum
    else:
        uniform_prob = 1 / len(pos_regret_dict)
        for action in policy.keys():
            policy[action] = uniform_prob


def regret_matching_plus(policy, regret_dict):
    pos_regret_sum = 0.0
    for action, regret in regret_dict.items():
        regret = max(0, regret)
        regret_dict[action] = regret
        pos_regret_sum += regret
    if pos_regret_sum > 0.0:
        for action, pos_regret in regret_dict.items():
            policy[action] = pos_regret / pos_regret_sum
    else:
        uniform_prob = 1 / len(regret_dict)
        for action in policy.keys():
            policy[action] = uniform_prob


def predictive_regret_matching(prediction, policy, regret_dict):
    pos_regret_dict = dict()
    pos_regret_sum = 0.0
    avg_prediction = sum(
        [prediction[action] * action_prob for action, action_prob in policy.items()]
    )
    for action, regret in regret_dict.items():
        pos_action_regret = max(0, regret + (prediction[action] - avg_prediction))
        pos_regret_dict[action] = pos_action_regret
        pos_regret_sum += pos_action_regret
    if pos_regret_sum > 0.0:
        for action, pos_regret in pos_regret_dict.items():
            policy[action] = pos_regret / pos_regret_sum
    else:
        uniform_prob = 1 / len(pos_regret_dict)
        for action in policy.keys():
            policy[action] = uniform_prob


def predictive_regret_matching_plus(prediction, policy, regret_dict):
    pos_regret_sum = 0.0
    avg_prediction = sum(
        [prediction[action] * action_prob for action, action_prob in policy.items()]
    )
    for action, regret in regret_dict.items():
        regret = max(0, regret)
        regret_dict[action] = regret
        pos_action_regret = max(0, regret + prediction[action] - avg_prediction)
        pos_regret_sum += pos_action_regret
    if pos_regret_sum > 0.0:
        for action, pos_regret in regret_dict.items():
            policy[action] = pos_regret / pos_regret_sum
    else:
        uniform_prob = 1 / len(regret_dict)
        for action in policy.keys():
            policy[action] = uniform_prob


def counterfactual_reach_prob(reach_prob_map: dict[int, float], player: int):
    return reduce(
        lambda x, y: x * y,
        [rp for i, rp in reach_prob_map.items() if i != player],
        1.0,
    )


def kuhn_optimal_policy(alpha: float):
    policy = dict()
    # player 0
    policy[0] = {
        "0": {0: 1 - alpha, 1: alpha},
        "0pb": {0: 1, 1: 0},
        "1": {0: 1, 1: 0},
        "1pb": {0: 2.0 / 3.0 - alpha, 1: 1.0 / 3.0 + alpha},
        "2": {0: 1 - 3.0 * alpha, 1: 3.0 * alpha},
        "2pb": {0: 0, 1: 1},
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
        root = (pyspiel.load_game(game) if isinstance(game, str) else game).new_initial_state()
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


class Player(Enum):
    alex = 0
    bob = 1
