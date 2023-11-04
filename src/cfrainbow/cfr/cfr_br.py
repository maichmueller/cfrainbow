from typing import Dict, Optional

import pyspiel

from cfrainbow.spiel_types import Action, Infostate, Player, Probability
from cfrainbow.utils import child_reach_prob_map, infostates_gen

from .cfr_vanilla import VanillaCFR, iterate_logging


class BestResponseCFR(VanillaCFR):
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
        self.infostate_to_player: Dict[Infostate, Player] = dict()

        for root_state in self.root_states:
            for infostate, player, state, _ in infostates_gen(root_state):
                self.infostate_to_player[infostate] = player
                self._set_action_list(infostate, state)
                self.regret_minimizer(infostate)

        self._best_response: Dict[Infostate, Action] = dict()

    @iterate_logging
    def iterate(
        self,
        updating_player: Optional[Player] = None,
    ):
        updating_player = self._cycle_updating_player(updating_player)

        # collect the entire policy of the updating player, so that we can compute the best response via openspiel
        player_current_policy = dict()
        for infostate, rm in self._regret_minimizer_dict.items():
            if self.infostate_to_player[infostate] == updating_player:
                player_current_policy[infostate] = list(
                    rm.recommend(self.iteration).items()
                )
        # let openspiel provide us with a dict that maps infostates to actions.
        # These actions will be the opponent's best response.
        self._best_response = pyspiel.TabularBestResponse(
            self.game,
            self._peek_at_next_updating_player(),
            player_current_policy,
        ).get_best_response_actions()
        # do a regular cfr iteration
        updating_player = self._cycle_updating_player(updating_player)
        for root_state, root_reach_prob_map in zip(
            self.root_states, self.root_reach_probabilities
        ):
            self._traverse(
                root_state.clone(),
                reach_prob_map=root_reach_prob_map.copy(),
                updating_player=updating_player,
            )

        # free the br dict memory, as it is no longer needed in this iteration
        self._best_response = None

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

        curr_player = state.current_player()
        infostate = state.information_state_string(curr_player)
        current_player = state.current_player()
        state_value = [0.0] * self.nr_players
        regret_minimizer = self.regret_minimizer(infostate)

        if current_player != updating_player:
            br_action = self._best_response[infostate]
            for action in regret_minimizer.actions:
                action_prob = 1.0 * (br_action == action)
                child_value = self._traverse(
                    state.child(action),
                    child_reach_prob_map(reach_prob_map, current_player, action_prob),
                    updating_player,
                )
                for p in self.players:
                    state_value[p] += action_prob * child_value[p]
            # no update to policy and regret map in this case
            return state_value

        else:
            action_values = dict()
            for action, action_prob in regret_minimizer.recommend(
                self.iteration
            ).items():
                child_value = self._traverse(
                    state.child(action),
                    child_reach_prob_map(reach_prob_map, current_player, action_prob),
                    updating_player,
                )
                action_values[action] = child_value
                for p in self.players:
                    state_value[p] += action_prob * child_value[p]

            self._update_regret_and_policy(
                curr_player,
                infostate,
                action_values,
                state_value,
                reach_prob_map,
            )

            return state_value
