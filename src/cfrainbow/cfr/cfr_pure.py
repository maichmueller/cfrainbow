from copy import deepcopy, copy
from typing import Optional, Dict, Mapping

from .cfr_base import CFRBase, iterate_logging
import pyspiel

from cfrainbow.utils import sample_on_policy, counterfactual_reach_prob
from cfrainbow.spiel_types import Action, Infostate, Probability


class PureCFR(CFRBase):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.plan: Dict[Infostate, Action] = {}

    @iterate_logging
    def iterate(
        self,
        updating_player: Optional[int] = None,
    ):
        # empty the previously sampled strategy
        self.plan.clear()

        self._traverse(
            self.root_state.clone(),
            reach_prob=(
                {player: 1.0 for player in self.players} if self.simultaneous else None
            ),
            updating_player=self._cycle_updating_player(updating_player),
        )
        self._iteration += 1

    def _traverse(
        self,
        state: pyspiel.State,
        reach_prob: Optional[Dict[int, Probability]] = None,
        updating_player: Optional[int] = None,
    ):
        self._nodes_touched += 1

        if state.is_terminal():
            return state.returns()

        if state.is_chance_node():
            return self._traverse_chance_node(state, reach_prob, updating_player)

        curr_player = state.current_player()
        infostate = state.information_state_string(curr_player)
        self._set_action_list(infostate, state)
        actions = self.action_list(infostate)
        regret_minimizer = self.regret_minimizer(infostate)
        player_policy = regret_minimizer.recommend(self.iteration)

        sampled_action = self._sample_action(infostate, player_policy)

        if self.simultaneous or curr_player == updating_player:
            if self.simultaneous:
                # update the average policy for the player
                self._avg_policy_at(curr_player, infostate)[sampled_action] += 1
                # TODO: Simultaneous updating PURE CFR only works if we actually multiply the counterfactual reach
                #  probability to the regret increment. Therefore, I am mixing the regret update rule from
                #  chance-sampling and pure cfr: The state value is computed according to pure-cfr's sampled
                #  action-value, but the difference of each action value to the state value is then multiplied by
                #  the cf. reach probability as in chance-sampling. Why this ends up being a correct regret update
                #  is unclear, even more so because the policy update is exactly according to pure cfr, and not
                #  chance-sampling.
                prob_weight = counterfactual_reach_prob(reach_prob, curr_player)
            else:
                prob_weight = 1.0

            action_values = dict()
            for action in actions:
                if self.simultaneous:
                    child_reach_prob = copy(reach_prob)
                    child_reach_prob[curr_player] *= player_policy[action]
                else:
                    child_reach_prob = reach_prob

                action_values[action] = self._traverse(
                    state.child(action),
                    child_reach_prob,
                    updating_player,
                )

            state_value = action_values[sampled_action]
            player_state_value = state_value[curr_player]

            regret_minimizer.observe(
                self.iteration,
                lambda a: prob_weight
                * (action_values[a][curr_player] - player_state_value),
            )
        else:
            if curr_player == self._peek_at_next_updating_player():
                self._avg_policy_at(curr_player, infostate)[sampled_action] += 1
            state.apply_action(sampled_action)
            state_value = self._traverse(state, reach_prob, updating_player)

        return state_value

    def _traverse_chance_node(self, state, reach_prob, updating_player):
        outcome, _, _ = sample_on_policy(
            *zip(*state.chance_outcomes()),
            rng=self.rng,
        )
        state.apply_action(int(outcome))
        return self._traverse(state, reach_prob, updating_player)

    def _avg_policy_at(self, current_player, infostate):
        if infostate not in (player_policy := self._avg_policy[current_player]):
            player_policy[infostate] = {
                action: 0.0 for action in self.action_list(infostate)
            }
        return player_policy[infostate]

    def _sample_action(
        self,
        infostate: Infostate,
        player_policy: Mapping[Action, Probability],
        *args,
        **kwargs,
    ):
        if infostate not in self.plan:
            actions = self.action_list(infostate)
            self.plan[infostate], _, _ = sample_on_policy(
                values=actions,
                policy=[player_policy[action] for action in actions],
                rng=self.rng,
            )
        return self.plan[infostate]
