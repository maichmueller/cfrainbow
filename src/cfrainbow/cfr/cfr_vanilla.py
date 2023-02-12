from copy import deepcopy
from typing import Dict, Mapping, Optional, Sequence

from .cfr_base import CFRBase, iterate_logging
from cfrainbow.spiel_types import Action, Infostate, Probability, Value

from cfrainbow.rm import ExternalRegretMinimizer
import pyspiel

from cfrainbow.utils import counterfactual_reach_prob


class CFRVanilla(CFRBase):
    @iterate_logging
    def iterate(
        self,
        updating_player: Optional[int] = None,
    ):
        self._traverse(
            self.root_state.clone(),
            reach_prob_map={player: 1.0 for player in [-1] + self.players},
            updating_player=self._cycle_updating_player(updating_player),
        )
        self._iteration += 1

    def _traverse(
        self,
        state: pyspiel.State,
        reach_prob_map: dict[Action, Probability],
        updating_player: Optional[int] = None,
    ):
        self._nodes_touched += 1

        if state.is_terminal():
            return state.returns()

        if state.is_chance_node():
            return self._traverse_chance_node(state, reach_prob_map, updating_player)
        else:
            curr_player = state.current_player()
            infostate = state.information_state_string(curr_player)
            self._set_action_list(infostate, state)
            action_values = self._action_value_map(infostate)
            state_value = self._traverse_player_node(
                state, infostate, reach_prob_map, updating_player, action_values
            )
            if self.simultaneous or updating_player == curr_player:
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
        state_value = [0.0] * self.nr_players
        for outcome, outcome_prob in state.chance_outcomes():
            next_state = state.child(outcome)

            child_reach_prob = deepcopy(reach_prob)
            child_reach_prob[state.current_player()] *= outcome_prob

            action_value = self._traverse(next_state, child_reach_prob, updating_player)
            for p in self.players:
                state_value[p] += outcome_prob * action_value[p]
        return state_value

    def _traverse_player_node(
        self, state, infostate, reach_prob, updating_player, action_values
    ):
        current_player = state.current_player()
        state_value = [0.0] * self.nr_players

        for action, action_prob in (
            self.regret_minimizer(infostate).recommend(self.iteration).items()
        ):
            child_reach_prob = deepcopy(reach_prob)
            child_reach_prob[current_player] *= action_prob
            next_state = state.child(action)

            child_value = self._traverse(next_state, child_reach_prob, updating_player)
            action_values[action] = child_value
            for p in self.players:
                state_value[p] += action_prob * child_value[p]

        return state_value

    def _update_regret(
        self,
        regret_minimizer: ExternalRegretMinimizer,
        action_values: Mapping[Action, Sequence[Value]],
        state_value: Sequence[Value],
        reach_probs: Dict[int, Probability],
        curr_player: int,
    ):
        player_state_value = (
            state_value[curr_player] if regret_minimizer.regret_mode else 0.0
        )
        cf_reach_p = counterfactual_reach_prob(reach_probs, curr_player)
        regret_minimizer.observe(
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
