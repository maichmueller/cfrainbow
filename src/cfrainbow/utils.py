from __future__ import annotations
import cmath
import inspect
import itertools
import operator
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from functools import reduce, singledispatchmethod
from typing import Dict, List, Union, Any, Sequence, Mapping, Optional, Tuple, Type

import numpy as np
import pyspiel

import threading

from cfrainbow.spiel_types import Infostate, Action, Probability, Player


def counterfactual_reach_prob(reach_prob_map: Mapping[int, float], player: int):
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
    root: Optional[pyspiel.State] = None,
    include_chance_states: bool = False,
    include_terminal_states: bool = False,
    *,
    depth: int = cmath.inf,
    game: Optional[Union[pyspiel.Game, str]] = None,
):
    if root is None:
        assert game is not None
        root = load_game(game).new_initial_state()
    stack = [(root, 0)]
    while stack:
        s, d = stack.pop()
        if s.is_terminal():
            if d < depth and include_terminal_states:
                yield s, d
            continue
        if s.is_chance_node():
            if include_chance_states:
                yield s, d
            for outcome, prob in s.chance_outcomes():
                if d < depth:
                    stack.append((s.child(outcome), d + 1))
        else:
            yield s, d
            for action in s.legal_actions():
                if d < depth:
                    stack.append((s.child(action), d + 1))


def terminal_states_gen(
    root: pyspiel.State,
):
    stack = [(root, 0)]
    while stack:
        s, d = stack.pop()
        if s.is_terminal():
            yield s, d
            continue
        for action in (
            s.legal_actions()
            if not s.is_chance_node()
            else (outcome for outcome, _ in s.chance_outcomes())
        ):
            stack.append((s.child(action), d + 1))


def infostates_gen(
    root,
):
    for state, d in all_states_gen(root, False, False):
        curr_player = state.current_player()
        yield state.information_state_string(curr_player), curr_player, state, d


def to_pyspiel_policy(
    policy_list,
    default_policy: Optional[
        Dict[
            Infostate,
            List[Tuple[Action, Probability]],
        ]
    ] = None,
) -> pyspiel.TabularPolicy:
    joint_policy = {
        istate: [
            (action, prob / max(1e-8, sum(as_and_ps.values())))
            for action, prob in as_and_ps.items()
        ]
        for istate, as_and_ps in itertools.chain(
            policy_list[0].items(), policy_list[1].items()
        )
    }
    if default_policy is not None:
        default_policy.update(joint_policy)
    else:
        default_policy = joint_policy
    return pyspiel.TabularPolicy(default_policy)


def sample_on_policy(
    values: Sequence[Any],
    policy: Sequence[float],
    rng: np.random.Generator,
    epsilon: float = 0.0,
):
    if epsilon != 0.0:
        uniform_prob = 1.0 / len(policy)
        policy = [
            epsilon * uniform_prob + (1 - epsilon) * policy_prob
            for policy_prob in policy
        ]
    choice = rng.choice(np.arange(len(values)), p=policy)
    return values[choice], choice, policy


ActionPolicyHint = Union[Dict[Action, Probability], List[Tuple[Action, Probability]]]
StatePolicyHint = Dict[Infostate, ActionPolicyHint]


def normalize_action_policy(action_policy: ActionPolicyHint):
    norm_action_policy = {}
    prob_sum = sum(action_policy.values())
    if prob_sum <= 0:
        raise ValueError(f"Policy values summed up to {prob_sum} <= 0.")
    for action, prob in action_policy.items():
        norm_action_policy[action] = prob / prob_sum
    return norm_action_policy


def normalize_state_policy(state_policy: StatePolicyHint):
    norm_policy = {}
    for infostate, action_policy in state_policy.items():
        norm_action_policy = normalize_action_policy(action_policy)
        norm_policy[infostate] = norm_action_policy
    return norm_policy


def normalize_policy_profile(
    policy_profile: Union[Dict[Player, StatePolicyHint], List[StatePolicyHint]]
):
    norm_policy_profile = dict()
    for player, state_policy in (
        {player: policy for player, policy in enumerate(policy_profile)}
        if isinstance(policy_profile, list)
        else policy_profile
    ).items():
        norm_action_policy = normalize_state_policy(state_policy)
        norm_policy_profile[player] = norm_action_policy
    return norm_policy_profile


class PolicyPrinter(ABC):
    register: Dict[pyspiel.Game, Type[PolicyPrinter]]

    def print_profile(
        self,
        policy_profile: Union[
            Dict[Player, Dict[Infostate, Dict[Action, Probability]]],
            Dict[Player, pyspiel.TabularPolicy],
        ],
    ) -> str:
        prints = []
        policy_profile = sorted(policy_profile.items(), key=operator.itemgetter(0))
        for player, policy in policy_profile:
            if isinstance(policy, pyspiel.TabularPolicy):
                policy = policy.policy_table()
            prints.append(f"PLAYER {player + 1}:\n{self.print_policy(player, policy)}")
        return "\n".join(prints)

    @abstractmethod
    def print_policy(
        self,
        player: Player,
        policy: Union[
            Dict[Infostate, Dict[Action, Probability]],
            pyspiel.TabularPolicy,
        ],
    ) -> str:
        raise NotImplementedError(
            f"{self.print_policy.__name__} has not been implemented."
        )


class EmptyPolicyPrinter(PolicyPrinter):
    def print_profile(self, *args, **kwargs) -> str:
        return ""

    def print_policy(self, *args, **kwargs) -> str:
        return ""


_fill_len = len(f"P1: {'?'.center(5, ' ')} | P2: {'?'.center(5, ' ')} | c b ")
_kuhn_poker_infostate_translation = {
    k: v.ljust(_fill_len, " ")
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


class PokerPolicyPrinter(PolicyPrinter):
    def __init__(self, digits: int = 3):
        self.digits = digits

    @classmethod
    @abstractmethod
    def action_name(cls, action: Action) -> str:
        raise NotImplementedError(
            f"{cls.action_name.__name__} has not been implemented."
        )

    def print_policy(
        self,
        player: Player,
        policy: Union[
            Dict[Infostate, Dict[Action, Probability]],
            pyspiel.TabularPolicy,
        ],
    ) -> str:
        if isinstance(policy, pyspiel.TabularPolicy):
            policy = policy.policy_table()

        out = []
        for infostate, action_policy in policy.items():
            action_policy = list(
                action_policy.items()
                if isinstance(action_policy, dict)
                else action_policy
            )
            out.append(
                f"{_kuhn_poker_infostate_translation[(infostate, player)]} "
                f"--> "
                f"{list(f'{self.action_name(action)}: {prob: .{self.digits}f}' for action, prob in action_policy)}"
            )
        return "\n".join(out)


class KuhnPolicyPrinter(PokerPolicyPrinter):
    class KuhnAction(Enum):
        check = 0
        bet = 1

    @classmethod
    def action_name(cls, action: Action) -> str:
        return cls.KuhnAction(action).name


class LeducPolicyPrinter(PokerPolicyPrinter):
    class LeducAction(Enum):
        Fold = 0
        Call = 1
        Raise = 2

    @classmethod
    def action_name(cls, action: Action) -> str:
        return cls.LeducAction(action).name


def slice_kwargs(given_kwargs, *func):
    possible_kwargs = set()
    for f in func:
        possible_kwargs = possible_kwargs.union(inspect.signature(f).parameters)
    return {k: v for k, v in given_kwargs.items() if k in possible_kwargs}


class SingletonMeta(type):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class KeyDependantDefaultDict(defaultdict):
    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def load_game(game: Union[pyspiel.Game, str]) -> pyspiel.Game:
    if isinstance(game, str):
        game = pyspiel.load_game(game)
    return game


def make_uniform_policy(root_state_or_game: Union[pyspiel.State, pyspiel.Game, str]):
    if isinstance(root_state_or_game, (pyspiel.Game, str)):
        root_state_or_game = load_game(root_state_or_game).new_initial_state()

    uniform_joint_policy = dict()
    for state, _ in all_states_gen(root=root_state_or_game):
        actions = state.legal_actions()
        uniform_joint_policy[state.information_state_string(state.current_player())] = [
            (action, 1.0 / len(actions)) for action in actions
        ]
    return uniform_joint_policy
