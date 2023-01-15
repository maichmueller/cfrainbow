from collections import defaultdict, deque
from copy import deepcopy
from typing import Dict, Mapping, Optional, Type, Sequence, MutableMapping
from type_aliases import Action, Infostate, Probability, Value
import numpy as np

import rm
from rm import ExternalRegretMinimizer
import pyspiel

from utils import (
    counterfactual_reach_prob,
    print_final_policy_profile,
    print_kuhn_poker_policy_profile,
    to_pyspiel_tab_policy,
    normalize_policy_profile,
)


class CFR2:
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
    ):
        self.root_state = root_state
        self.players = list(range(root_state.num_players()))
        self.nr_players = len(self.players)
        self.regret_minimizer_type: Type[
            ExternalRegretMinimizer
        ] = regret_minimizer_type
        self._regret_minimizer_dict: Dict[Infostate, ExternalRegretMinimizer] = {}
        self._avg_policy = (
            average_policy_list
            if average_policy_list is not None
            else [{} for _ in self.players]
        )
        self._action_set: Dict[Infostate, Sequence[Action]] = {}
        self._player_update_cycle = deque(self.players)
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

    def average_policy(self, player: Optional[int] = None):
        if player is None:
            return self._avg_policy
        else:
            return [self._avg_policy[player]]

    def iterate(
        self,
        traversing_player: Optional[int] = None,
    ):
        traversing_player = self._cycle_updating_player(traversing_player)

        if self._verbose:
            print(
                "\nIteration",
                self._alternating_update_msg()
                if self.alternating
                else self.iteration
            )
        self._traverse(
            self.root_state.clone(),
            reach_prob_map={player: 1.0 for player in [-1] + self.players},
            traversing_player=traversing_player,
        )
        self._iteration += 1

    def _alternating_update_msg(self):
        divisor, remainder = divmod(self.iteration, self.nr_players)
        # '[iteration] [player] / [nr_players]' to highlight which player of this update cycle is currently updated
        return f"{divisor} {(remainder + 1)}/{self.nr_players}"

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

    def regret_minimizer(self, infostate: Infostate):
        if infostate not in self._regret_minimizer_dict:
            self._regret_minimizer_dict[infostate] = self.regret_minimizer_type(
                self.action_list(infostate)
            )
        return self._regret_minimizer_dict[infostate]

    def _avg_policy_at(self, current_player, infostate):
        if infostate not in (player_policy := self._avg_policy[current_player]):
            player_policy[infostate] = defaultdict(float)
        return player_policy[infostate]

    def _set_action_list(self, infostate: Infostate, state: pyspiel.State):
        if infostate not in self._action_set:
            self._action_set[infostate] = state.legal_actions()

    def action_list(self, infostate: Infostate):
        if infostate not in self._action_set:
            raise KeyError(f"Infostate {infostate} not in action list lookup.")
        return self._action_set[infostate]

    def _action_value_map(self, infostate: Infostate):
        return dict()

    def _traverse(
        self,
        state: pyspiel.State,
        reach_prob_map: dict[Action, Probability],
        traversing_player: Optional[int] = None,
    ):
        if state.is_terminal():
            return state.returns()

        if state.is_chance_node():
            return self._traverse_chance_node(
                state, reach_prob_map, traversing_player
            )
        else:
            curr_player = state.current_player()
            infostate = state.information_state_string(curr_player)

            action_values = self._action_value_map(infostate)
            state_value = self._traverse_player_node(
                state, infostate, reach_prob_map, traversing_player, action_values
            )
            if self.simultaneous or traversing_player == curr_player:
                regret_minimizer = self.regret_minimizer(infostate)
                self._update_regret(
                    regret_minimizer,
                    action_values,
                    state_value,
                    reach_prob_map,
                    curr_player,
                )
                self._update_avg_policy(
                    regret_minimizer.recommend(self.iteration),
                    reach_prob_map,
                    infostate,
                    curr_player,
                )
            return state_value

    def _traverse_chance_node(self, state, reach_prob, updating_player):
        action_values = {}
        state_value = np.zeros(len(self.players))
        for outcome, outcome_prob in state.chance_outcomes():
            next_state = state.child(outcome)

            child_reach_prob = deepcopy(reach_prob)
            child_reach_prob[state.current_player()] *= outcome_prob

            action_values[outcome] = self._traverse(
                next_state, child_reach_prob, updating_player
            )
            state_value += outcome_prob * np.asarray(action_values[outcome])
        return state_value

    def _traverse_player_node(
        self, state, infostate, reach_prob, updating_player, action_values
    ):
        current_player = state.current_player()
        state_value = np.zeros(len(self.players))

        self._set_action_list(infostate, state)
        regret_minimizer = self.regret_minimizer(infostate)
        current_policy = regret_minimizer.recommend(self.iteration)

        for action, action_prob in current_policy.items():
            child_reach_prob = deepcopy(reach_prob)
            child_reach_prob[current_player] *= action_prob
            next_state = state.child(action)

            action_values[action] = self._traverse(
                next_state, child_reach_prob, updating_player
            )
            state_value += action_prob * np.asarray(action_values[action])

        return state_value

    def _update_regret(
        self,
        regret_minimizer: ExternalRegretMinimizer,
        action_values: Mapping[Action, Sequence[Value]],
        state_value: Sequence[Value],
        reach_probs: Dict[int, Probability],
        curr_player: int,
    ):
        player_state_value = state_value[curr_player]
        cf_reach_p = counterfactual_reach_prob(reach_probs, curr_player)
        regret_minimizer.observe_regret(
            self.iteration,
            lambda a: cf_reach_p * (action_values[a][curr_player] - player_state_value),
        )

    def _update_avg_policy(
        self,
        curr_policy: Dict[Action, Probability],
        reach_prob: Dict[int, Probability],
        infostate: Infostate,
        curr_player: int,
    ):
        player_reach_prob = reach_prob[curr_player]
        avg_policy = self._avg_policy_at(curr_player, infostate)
        for action, curr_policy_prob in curr_policy.items():
            avg_policy[action] += player_reach_prob * curr_policy_prob



def main(n_iter, do_print: bool = True):
    from open_spiel.python.algorithms import exploitability

    if do_print:
        print(f"Running CFR with alternating updates for {n_iter} iterations.")
    expl_values = []
    game = pyspiel.load_game("kuhn_poker")
    root_state = game.new_initial_state()
    n_players = list(range(root_state.num_players()))
    average_policies = [{} for _ in n_players]
    solver = CFR2(
        root_state,
        regret_minimizer_type=rm.RegretMatcher,
        average_policy_list=average_policies,
        verbose=do_print,
    )
    for i in range(n_iter):
        solver.iterate()

        expl_values.append(
            exploitability.exploitability(
                game,
                to_pyspiel_tab_policy(average_policies),
            )
        )

        if do_print:
            print(
                f"-------------------------------------------------------------"
                f"--> Exploitability {expl_values[-1]: .5f}"
            )
            print_kuhn_poker_policy_profile(normalize_policy_profile(average_policies))
            print(f"---------------------------------------------------------------")
    if do_print:
        print_final_policy_profile(average_policies)

    return expl_values


if __name__ == "__main__":
    main(n_iter=2000, do_print=True)
