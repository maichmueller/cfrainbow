from typing import Optional, Dict

import pyspiel

from cfrainbow.spiel_types import Action, Probability, Player, Infostate
from cfrainbow.utils import counterfactual_reach_prob, infostates_gen
from .cfr_base import CFRBase, iterate_logging


class CFRBestResponse(CFRBase):
    def __init__(
        self,
        *args,
        alpha=float("inf"),
        beta=-float("inf"),
        gamma=1,
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
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            **kwargs,
        )
        self.infostate_to_player: Dict[Infostate, Player] = dict()

        for infostate, player, state, _ in infostates_gen(self.root_state.clone()):
            self.infostate_to_player[infostate] = player
            self._set_action_list(infostate, state)
            self.regret_minimizer(infostate)

    @iterate_logging
    def iterate(
        self,
        updating_player: Optional[Player] = None,
    ):
        updating_player = self._cycle_updating_player(updating_player)

        player_current_policy = dict()
        for infostate, rm in self._regret_minimizer_dict.items():
            if self.infostate_to_player[infostate] == updating_player:
                player_current_policy[infostate] = list(rm.recommend(self.iteration).items())
        br = pyspiel.TabularBestResponse(
            self.root_state.get_game(),
            self._peek_at_next_updating_player(),
            player_current_policy,
        ).get_best_response_actions()

        self._traverse(
            self.root_state.clone(),
            reach_prob_map={player: 1.0 for player in [-1] + self.players},
            updating_player=updating_player,
            best_response=br,
        )

        self._iteration += 1

    def _traverse(
        self,
        state: pyspiel.State,
        reach_prob_map: Dict[Action, Probability],
        updating_player: Player,
        best_response: Dict[Infostate, Action],
    ):
        self._nodes_touched += 1

        if state.is_terminal():
            return state.returns()

        if state.is_chance_node():
            return self._traverse_chance_node(
                state, reach_prob_map, updating_player, best_response
            )
        else:
            curr_player = state.current_player()
            infostate = state.information_state_string(curr_player)
            action_values = self._action_value_map(infostate)
        current_player = state.current_player()
        state_value = [0.0] * self.nr_players
        regret_minimizer = self.regret_minimizer(infostate)

        if current_player != updating_player:
            br_action = best_response[infostate]
            for action in regret_minimizer.actions:
                action_prob = 1.0 * (br_action == action)
                child_value = self._traverse(
                    state.child(action),
                    self.child_reach_prob_map(
                        reach_prob_map, current_player, action_prob
                    ),
                    updating_player,
                    best_response,
                )
                action_values[action] = child_value
                for p in self.players:
                    state_value[p] += action_prob * child_value[p]

            return state_value

        for action, action_prob in regret_minimizer.recommend(self.iteration).items():
            child_value = self._traverse(
                state.child(action),
                self.child_reach_prob_map(
                    reach_prob_map, current_player, action_prob
                ),
                updating_player,
                best_response,
            )
            action_values[action] = child_value
            for p in self.players:
                state_value[p] += action_prob * child_value[p]

        # update the cumulative regret
        player_state_value = (
            state_value[curr_player]
            if regret_minimizer.observes_regret
            else 0.0
        )
        cf_reach_p = counterfactual_reach_prob(reach_prob_map, curr_player)
        regret_minimizer.observe(
            self.iteration,
            lambda a: cf_reach_p
            * (action_values[a][curr_player] - player_state_value),
        )
        # update the average policy
        player_reach_prob = reach_prob_map[curr_player]
        avg_policy = self._avg_policy_at(curr_player, infostate)
        for action, curr_policy_prob in regret_minimizer.recommend(
            self.iteration
        ).items():
            avg_policy[action] += player_reach_prob * curr_policy_prob

        return state_value

    def _traverse_chance_node(self, state, reach_prob, updating_player, best_response):
        state_value = [0.0] * self.nr_players
        for outcome, outcome_prob in state.chance_outcomes():
            action_value = self._traverse(
                state.child(outcome),
                self.child_reach_prob_map(
                    reach_prob, state.current_player(), outcome_prob
                ),
                updating_player,
                best_response=best_response,
            )
            for p in self.players:
                state_value[p] += outcome_prob * action_value[p]
        return state_value
