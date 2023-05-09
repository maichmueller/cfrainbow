from typing import Optional, Dict

import pyspiel

from cfrainbow.spiel_types import Action, Probability, Player
from cfrainbow.utils import counterfactual_reach_prob
from .cfr_base import CFRBase, iterate_logging


class VanillaCFR(CFRBase):
    @iterate_logging
    def iterate(
        self,
        updating_player: Optional[Player] = None,
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
        reach_prob_map: Dict[Action, Probability],
        updating_player: Optional[Player] = None,
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
            action_values = dict()
            state_value = self._traverse_player_node(
                state, infostate, reach_prob_map, updating_player, action_values
            )
            if self.simultaneous or updating_player == curr_player:
                self._update_regret_and_policy(
                    curr_player,
                    infostate,
                    action_values,
                    state_value,
                    reach_prob_map,
                )
            return state_value

    def _traverse_chance_node(self, state, reach_prob, updating_player):
        state_value = [0.0] * self.nr_players
        for outcome, outcome_prob in state.chance_outcomes():
            action_value = self._traverse(
                state.child(outcome),
                self.child_reach_prob_map(
                    reach_prob, state.current_player(), outcome_prob
                ),
                updating_player,
            )
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
            child_value = self._traverse(
                state.child(action),
                self.child_reach_prob_map(reach_prob, current_player, action_prob),
                updating_player,
            )
            action_values[action] = child_value
            for p in self.players:
                state_value[p] += action_prob * child_value[p]

        return state_value

    def _update_regret_and_policy(
        self,
        curr_player,
        infostate,
        action_values,
        state_value,
        reach_prob_map,
    ):
        regret_minimizer = self.regret_minimizer(infostate)
        # update the cumulative regret
        player_state_value = (
            state_value[curr_player] if regret_minimizer.observes_regret else 0.0
        )
        cf_reach_p = counterfactual_reach_prob(reach_prob_map, curr_player)
        regret_minimizer.observe(
            self.iteration,
            lambda a: cf_reach_p * (action_values[a][curr_player] - player_state_value),
        )
        # update the average policy
        player_reach_prob = reach_prob_map[curr_player]
        avg_policy = self._avg_policy_at(curr_player, infostate)
        for action, curr_policy_prob in regret_minimizer.recommend(
            self.iteration
        ).items():
            avg_policy[action] += player_reach_prob * curr_policy_prob
