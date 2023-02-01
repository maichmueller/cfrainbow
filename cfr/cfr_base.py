import functools
import inspect
from collections import deque
from typing import Dict, Optional, Type, Sequence, MutableMapping, Union

import numpy as np
import pyspiel

from rm import ExternalRegretMinimizer
from spiel_types import Action, Infostate, Probability, Player


@functools.wraps
def iterate_log_print(f):
    def wrapped(self, *args, **kwargs):
        if self._verbose:
            print(
                "\nIteration",
                self._alternating_update_msg() if self.alternating else self.iteration,
            )
        return f(self, *args, **kwargs)

    return wrapped


class CFRBase:
    def __init__(
        self,
        root_state: pyspiel.State,
        regret_minimizer_type: Type[ExternalRegretMinimizer],
        *,
        average_policy_list: Optional[
            Sequence[MutableMapping[Infostate, MutableMapping[Action, Probability]]]
        ] = None,
        alternating: bool = True,
        verbose: bool = False,
        seed: Optional[Union[int, np.random.Generator]] = None,
        **regret_minimizer_kwargs,
    ):
        self.root_state = root_state
        self.players = list(range(root_state.num_players()))
        self.nr_players = len(self.players)
        self.regret_minimizer_type: Type[
            ExternalRegretMinimizer
        ] = regret_minimizer_type
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._regret_minimizer_dict: Dict[Infostate, ExternalRegretMinimizer] = {}
        self._regret_minimizer_kwargs = {
            k: v
            for k, v in regret_minimizer_kwargs.items()
            if k in inspect.signature(regret_minimizer_type.__init__).parameters
        }
        self._avg_policy = (
            average_policy_list
            if average_policy_list is not None
            else [{} for _ in self.players]
        )
        self._action_set: Dict[Infostate, Sequence[Action]] = {}
        self._player_update_cycle = deque(reversed(self.players))
        self._iteration = 0
        self._alternating = alternating
        self._verbose = verbose

    @property
    def iteration(self):
        return self._iteration

    @property
    def alternating(self):
        return self._alternating

    @property
    def simultaneous(self):
        return not self._alternating

    def average_policy(self, player: Optional[Player] = None):
        if player is None:
            return self._avg_policy
        else:
            return [self._avg_policy[player]]

    def regret_minimizer(self, infostate: Infostate):
        if infostate not in self._regret_minimizer_dict:
            self._regret_minimizer_dict[infostate] = self.regret_minimizer_type(
                self.action_list(infostate), **self._regret_minimizer_kwargs
            )
        return self._regret_minimizer_dict[infostate]

    def action_list(self, infostate: Infostate):
        if infostate not in self._action_set:
            raise KeyError(f"Infostate {infostate} not in action list lookup.")
        return self._action_set[infostate]

    def _set_action_list(self, infostate: Infostate, state: pyspiel.State):
        if infostate not in self._action_set:
            self._action_set[infostate] = state.legal_actions()

    def _cycle_updating_player(self, updating_player: Optional[int]):
        if self.simultaneous:
            return None
        if updating_player is None:
            # get the next updating player from the queue. This value will be returned
            updating_player = self._player_update_cycle.pop()
            # ...and emplace it back at the end of the queue
            self._player_update_cycle.appendleft(updating_player)
        else:
            # an updating player was forced from the outside, so move that player to the end of the update list
            self._player_update_cycle.remove(updating_player)
            self._player_update_cycle.appendleft(updating_player)
        return updating_player

    def _peek_at_next_updating_player(self):
        return self._player_update_cycle[-1]

    def _avg_policy_at(self, current_player, infostate):
        if infostate not in (player_policy := self._avg_policy[current_player]):
            player_policy[infostate] = {
                action: 0.0 for action in self.action_list(infostate)
            }
        return player_policy[infostate]

    def _alternating_update_msg(self):
        divisor, remainder = divmod(self.iteration, self.nr_players)
        # '[iteration] [player] / [nr_players]' to highlight which player of this update cycle is currently updated
        return f"{divisor} {(remainder + 1)}/{self.nr_players}"

    def _action_value_map(self, infostate: Optional[Infostate] = None):
        return dict()
