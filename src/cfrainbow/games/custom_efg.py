from __future__ import annotations
from copy import copy
from typing import Sequence, Mapping, Tuple, Union, Dict

import pyspiel

from cfrainbow.spiel_types import Player, Action


class Node:
    def __init__(self, name: str, children: Dict[str, str] = None, parent: str = None):
        self._name = name
        self._parent = parent
        self._children = dict(children)
        for i, (key, value) in enumerate(children.items()):
            # add integers to the possible keys list
            self._children[i] = value

    @property
    def name(self):
        return self._name

    @property
    def parent(self):
        return self._parent

    @property
    def children(self):
        return self._children

    def __hash__(self):
        return hash(self._name)

    def __contains__(self, item: Node):
        return item in self._children

    def __getitem__(self, action: Union[str, int]):
        return self._children[action]


class GameTree:
    def __init__(self, nodes: Sequence[Node]):
        self.nodes = {node.name: node for node in nodes}


_NUM_PLAYERS = 2
_GAME_TYPE = pyspiel.GameType(
    short_name="python_efce_example_efg",
    long_name="Python Basic EFG",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=False,
    provides_observation_tensor=False,
    provides_factored_observation_string=False,
)
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=2,
    num_players=_NUM_PLAYERS,
    max_chance_outcomes=0,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=3,
)
_GAME_TYPE_GENERAL_SUM = pyspiel.GameType(
    short_name="python_efce_example_efg_general_sum",
    long_name="Python Basic EFG General Sum",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=False,
    provides_observation_string=False,
    provides_observation_tensor=False,
    provides_factored_observation_string=False,
)
_GAME_INFO_GENERAL_SUM = pyspiel.GameInfo(
    num_distinct_actions=2,
    num_players=_NUM_PLAYERS,
    max_chance_outcomes=0,
    min_utility=-2.0,
    max_utility=2.0,
    utility_sum=0.0,
    max_game_length=3,
)

tree = GameTree(
    nodes=[
        Node("A", dict(a="X", b="Y")),
        Node("X", dict(x="B", v="C"), parent="A"),
        Node("Y", dict(y="D1", w="D2"), parent="A"),
        Node("B", dict(c="Z1", d="Z2"), parent="X"),
        Node("C", dict(e="Z3", f="Z4"), parent="X"),
        Node("D1", dict(g="Z5", h="Z7"), parent="Y"),
        Node("D2", dict(g="Z6", h="Z8"), parent="Z"),
    ]
    + [
        Node(f"Z{i}", dict(), p)
        for i, p in zip(range(1, 9), ["B", "B", "C", "C", "D1", "D1", "D2", "D2"])
    ]
)

default_payoffs = {
    "Z1": (1.0, 2.0),
    "Z2": (-1.0, 0.0),
    "Z3": (3.0, 1.0),
    "Z4": (-2.0, 4.0),
    "Z5": (5.0, 0.0),
    "Z6": (2.0, 2.0),
    "Z7": (1.0, 5.0),
    "Z8": (-4.0, 6.0),
}

zero_sum_payoffs = {
    "Z1": (1.0, -1.0),
    "Z2": (-1.0, 1.0),
    "Z3": (0.0, 0.0),
    "Z4": (1, -1.0),
    "Z5": (0.0, 0.0),
    "Z6": (0.0, 0.0),
    "Z7": (1.0, -1.0),
    "Z8": (-1.0, 1.0),
}


class GameZeroSum(pyspiel.Game):
    def __init__(self, params=None):
        super().__init__(
            _GAME_TYPE,
            _GAME_INFO,
            params or dict(),
        )

    def new_initial_state(self):
        return State(self, payoffs=zero_sum_payoffs)

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        return CustomEFGObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=True), params
        )


class GameGeneralSum(pyspiel.Game):
    def __init__(self, params=None):
        super().__init__(
            _GAME_TYPE_GENERAL_SUM, _GAME_INFO_GENERAL_SUM, params or dict()
        )

    def new_initial_state(self):
        return State(
            self,
            payoffs=default_payoffs,
        )

    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        return CustomEFGObserver(
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=True), params
        )


class State(pyspiel.State):
    def __init__(self, game: GameZeroSum, payoffs: Mapping[str, Tuple[float, float]]):
        super().__init__(game)
        self._node = tree.nodes["A"]
        self._payoffs = payoffs

    @property
    def node(self):
        return self._node

    @property
    def payoffs(self):
        return self._payoffs

    def __str__(self):
        return self.node.name

    def _apply_action(self, action: Action):
        self._node = tree.nodes[self.node[action]]

    def _legal_actions(self, player):
        return [0, 1]

    def is_terminal(self):
        return self.node.name.startswith("Z")

    def is_chance_node(self):
        return False

    def is_player_node(self):
        return True

    def current_player(self):
        if self.node.name.startswith("Z"):
            return pyspiel.PlayerId.TERMINAL
        if self.node.name in ["X", "Y"]:
            return 1
        else:
            return 0

    def returns(self):
        return self.payoffs[self.node.name]

    def player_return(self, player: Player):
        return self.payoffs[self.node.name][player]


class CustomEFGObserver:
    """Observer, conforming to the PyObserver interface (see observation.py)."""

    def __init__(self, iig_obs_type, params):
        """Initializes an empty observation tensor."""
        if params:
            raise ValueError(f"Observation parameters not supported; passed {params}")

        self.tensor = [0.0] * len(tree.nodes)
        self.dict = {}

    def set_from(self, state, player):
        """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
        pass

    @staticmethod
    def string_from(state, player):
        """Observation of `state` from the PoV of `player`, as a string."""
        if player == 0 and state.node.name.startswith("D"):
            return "D"
        else:
            return state.node.name
