import dataclasses
from typing import Dict, List, Optional

import pyspiel

from ..spiel_types import Infostate
from .cfr_base import CFRBase


class CFRLazy(CFRBase):
    """
    Following https://arxiv.org/pdf/1810.04433.pdf
    """

    @dataclasses.dataclass
    class RunningBelief:
        """
        Dataclass storing the `m1` and `m2` variables in the pseudocode of an infostate.

        Each variable is a list of running strategy contributions of the opponents to reach that infostate.
        Both values get reset at certain stages in the algorithm.
        """

        m1: List[float]
        m2: List[float]

        def __init__(self, nr_underlying_histories: int):
            self.m1 = [0.0] * nr_underlying_histories
            self.m2 = [0.0] * nr_underlying_histories

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        kwargs.update(
            dict(
                # only allow alternating update cycles
                alternating=True,
            )
        )
        super().__init__(
            *args,
            **kwargs,
        )
        self.infostate_trees = {
            player: pyspiel.InfostateTree(self.root_state.get_game(), player)
            for player in self.players
        }
        self.underlying_histories: Dict[Infostate, pyspiel.State] = dict()
        self.running_belief: Dict[Infostate, CFRLazy.RunningBelief] = dict()

    def _align_histories_and_infostates(self):
        infostate_to_infonodes = Dict[Infostate, pyspiel.InfostateNode] = dict()
        for player, infostate_tree in self.infostate_trees.items():
            stack = [infostate_tree.root()]
            while stack:
                node = stack.pop()
                if not node.is_filler_node():
                    if node.has_infostate_string():
                        print(node.infostate_string())
                        print(node.type())
                node = node.child_at(0)

    def iterate(self, updating_player: Optional[int] = None):
        updating_player = self._cycle_updating_player(updating_player)
        self._traverse_infotree(
            self.infostate_tree.root(),
        )
        self._traverse(
            self.root_state.clone(),
            reach_prob_map={player: 1.0 for player in [-1] + self.players},
            updating_player=updating_player,
        )

        self._iteration += 1
