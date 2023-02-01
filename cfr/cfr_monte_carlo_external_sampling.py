from copy import deepcopy
from typing import Optional

import pyspiel

from utils import sample_on_policy
from .cfr_base import CFRBase, iterate_log_print


class ExternalSamplingMCCFR(CFRBase):
    @iterate_log_print
    def iterate(self, traversing_player: Optional[int] = None):
        value = self._traverse(
            deepcopy(self.root_state),
            traversing_player=self._cycle_updating_player(traversing_player),
        )
        self._iteration += 1
        return value

    def _traverse(self, state: pyspiel.State, traversing_player: int = 0):
        current_player = state.current_player()
        if state.is_terminal():
            reward = state.player_return(traversing_player)
            return reward

        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            outcome, outcome_prob = self.rng.choice(outcomes)
            state.apply_action(int(outcome))

            return self._traverse(state, traversing_player)

        curr_player = state.current_player()
        infostate = state.information_state_string(curr_player)
        self._set_action_list(infostate, state)
        actions = self.action_list(infostate)
        regret_minimizer = self.regret_minimizer(infostate)
        player_policy = regret_minimizer.recommend(self.iteration)

        if traversing_player == current_player:
            state_value = 0.0
            action_values = dict()

            for action in actions:
                action_values[action] = self._traverse(
                    state.child(action), traversing_player
                )
                state_value += player_policy[action] * action_values[action]

            regret_minimizer.observe_regret(
                self.iteration, lambda a: action_values[a] - state_value
            )
            return state_value
        else:
            sampled_action, _, _ = sample_on_policy(
                values=actions,
                policy=[player_policy[action] for action in actions],
                rng=self.rng,
            )
            state.apply_action(sampled_action)
            action_value = self._traverse(state, traversing_player)

            if current_player == self._peek_at_next_updating_player():
                avg_policy = self._avg_policy_at(current_player, infostate)
                for action, prob in player_policy.items():
                    avg_policy[action] += prob

            return action_value
