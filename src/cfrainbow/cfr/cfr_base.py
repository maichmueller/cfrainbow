import functools
import inspect
from collections import deque
from copy import copy
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Type, Union

import numpy as np
import pyspiel

from cfrainbow.rm import ExternalRegretMinimizer
from cfrainbow.spiel_types import Action, Infostate, Player, Probability
from cfrainbow.utils import slice_kwargs, ChancePlayer


def iterate_logging(f):
    @functools.wraps(f)
    def wrapped(self, *args, **kwargs):
        if self.verbose:
            print(
                "\nIteration",
                self._alternating_update_msg() if self.alternating else self.iteration,
                f"\t Nodes Touched: {self.nodes_touched}",
            )
        return f(self, *args, **kwargs)

    return wrapped


def _assert_game_constraints(game: pyspiel.Game):
    game_type = game.get_type()
    assert (
        game_type.dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL,
        "CFR operates only on turn-based games.",
    )
    assert (
        game_type.chance_mode
        in (
            pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
            pyspiel.GameType.ChanceMode.DETERMINISTIC,
        ),
        "CFR operates only on discretely stochastic or deterministic games.",
    )


class CFRBase:
    """
    Base class for the Counterfactual Regret Minimization (CFR) algorithm.

    Parameters
    ----------
    root_state : pyspiel.State
        The root state of the game.
    regret_minimizer_type : Type[ExternalRegretMinimizer]
        The type of external regret minimizer to use.
    average_policy_list : Optional[Sequence[MutableMapping[Infostate, MutableMapping[Action, Probability]]]]
        The list of average policies per player, by default None.
    alternating : bool
        Whether to use alternating updates or not, by default True.
    verbose : bool
        Whether to print verbose output during the iterations, by default False.
    seed : Optional[Union[int, np.random.Generator]]
        The seed for the random number generator, by default None.
    **regret_minimizer_kwargs
        Keyword arguments to be passed to the external regret minimizer constructor.

    Attributes
    ----------
    iteration : int
        The current iteration of the CFR algorithm.
    alternating : bool
        Whether alternating updates are used or not.
    simultaneous : bool
        Whether simultaneous updates are used or not.
    nodes_touched : int
        The number of nodes touched during the CFR iterations.

    Methods
    -------
    iterate_for(n: int)
        Convenience wrapper to run n iterations of the algorithm.
    iterate(Optional[int] = None)
        Runs a single iteration of the CFR algorithm.
    average_policy(Optional[Player] = None)
        Fetches the average policies of the given player or all players.
    regret_minimizer(Infostate)
        Fetches the local regret minimizer for the given infostate.
    action_list(Infostate)
        Fetches the list of legal actions for the given infostate.
    force_update()
        Forces all regret minimizers to compute the latest recommendation.
    """

    def __init__(
        self,
        root_states: Union[Iterable[pyspiel.State], pyspiel.State],
        regret_minimizer_type: Type[ExternalRegretMinimizer],
        *,
        root_reach_probabilities: Optional[List[Dict[Player, Probability]]] = None,
        average_policy_list: Optional[
            Sequence[MutableMapping[Infostate, MutableMapping[Action, Probability]]]
        ] = None,
        alternating: bool = True,
        verbose: bool = False,
        seed: Optional[Union[int, np.random.Generator]] = None,
        **regret_minimizer_kwargs,
    ):
        if not isinstance(root_states, list):
            root_states = (root_states,)
        game: pyspiel.Game = root_states[0].get_game()
        _assert_game_constraints(game)
        players = list(range(game.num_players()))
        if root_reach_probabilities is not None:
            assert len(root_reach_probabilities) == len(
                root_states
            ), "Number of root reach probability maps does not match the number of passed root states"
        else:
            # if no reach probabilities are given then we expect the states to be true roots of the game
            root_reach_probabilities = [
                {player: Probability(1.0) for player in [ChancePlayer] + players}
                for _ in root_states
            ]
        # the passed parameters meet the constraints, so we continue to fill the members
        self.game = game
        self.root_states = root_states
        self.players = players
        self.root_reach_probabilities = root_reach_probabilities
        self.nr_players = len(self.players)
        self.regret_minimizer_type: Type[
            ExternalRegretMinimizer
        ] = regret_minimizer_type
        self.verbose = verbose
        self.seed: Optional[Union[int, np.random.Generator]] = seed
        self.rng = np.random.default_rng(seed)
        # the association of infostates to their local external regret minimizers.
        # This dict will be filled on demand once the infostates have been encountered.
        self._regret_minimizer_dict: Dict[Infostate, ExternalRegretMinimizer] = {}
        # these kwargs are going to be the keyword arguments with which every new regret minimizer
        # is going to be instantiated.
        self._regret_minimizer_kwargs = {
            k: v
            for k, v in slice_kwargs(
                regret_minimizer_kwargs, regret_minimizer_type.__init__
            ).items()
        }
        # the list of average policies per player
        self._avg_policy = (
            average_policy_list
            if average_policy_list is not None
            else [{} for _ in self.players]
        )
        self._action_set: Dict[Infostate, list[Action]] = {}
        # the update cycle for alternating update schemes. The default is 0 --> 1 --> 2 ...--> N --> 0 -->...
        self._player_update_cycle = deque(reversed(self.players))
        self._alternating = alternating
        self._iteration = 0
        self._nodes_touched = 0

    @property
    def iteration(self):
        return self._iteration

    @property
    def alternating(self):
        return self._alternating

    @property
    def simultaneous(self):
        return not self._alternating

    @property
    def nodes_touched(self):
        return self._nodes_touched

    def iterate_for(self, n: int):
        """
        Convenience wrapper to run n iterations of the algorithm.
        """
        for _ in range(n):
            self.iterate()

    def iterate(self, updating_player: Optional[int] = None):
        """
        The actual CFR algorithm implementation.

        Parameters
        ----------
        updating_player: Optional[int]
            an external force to update this specific player next. Default is None, which allows the normal update
            cycle to run. Note that if one passes a player in then the update cycle is permanently modified.

        Returns
        -------
        None
        """
        raise NotImplementedError(f"method {self.iterate.__name__} is not implemented.")

    def average_policy(self, player: Optional[Player] = None):
        """
        Fetch the average policies of the given player or all at once.

        Parameters
        ----------
        player: int
            the optional player whose average policy to fetch. If none, all player policies are returned.

        Returns
        -------
        List[Dict[str, Dict[int, float]
            a list of state policies (i.e. maps of infostates to action policies).
        """
        if player is None:
            return self._avg_policy
        else:
            return [self._avg_policy[player]]

    def regret_minimizer(self, infostate: Infostate):
        """
        Fetch the local regret minimizer in use at the given infostate.

        Parameters
        ----------
        infostate: str

        Returns
        -------
        ExternalRegretMinimizer
            the external regret minimizer minimizing the local regret at the infostate.
        """
        if infostate not in self._regret_minimizer_dict:
            self._regret_minimizer_dict[infostate] = self.regret_minimizer_type(
                self.action_list(infostate), **self._regret_minimizer_kwargs
            )
        return self._regret_minimizer_dict[infostate]

    def action_list(self, infostate: Infostate):
        """
        Fetch the list of legal actions at the infostate
        Parameters
        ----------
        infostate: str

        Returns
        -------
        List[int]
            legal actions available to the active player at the infostate.
        """
        if infostate not in self._action_set:
            raise KeyError(f"Infostate {infostate} not in action list lookup.")
        return self._action_set[infostate]

    def force_update(self):
        """
        Force all regret minimizers to compute the latest recommendation.
        """
        for regret_minimizer in self._regret_minimizer_dict.values():
            regret_minimizer.recommend(self.iteration, force=True)

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
        return f"{self.iteration} [{divisor} | {(remainder + 1)}/{self.nr_players}]"
