from collections import defaultdict, deque
from copy import deepcopy
from typing import Optional, Dict, Sequence, Union, Type, MutableMapping, Mapping

from cfr2 import CFR2
from cfr2_base import CFRBase
from rm import regret_matching, ExternalRegretMinimizer
import pyspiel
import numpy as np

from utils import sample_on_policy, counterfactual_reach_prob
from type_aliases import Action, Infostate, Probability, Regret, Value


class PureCFR2(CFRBase):
    def __init__(
        self,
        *args,
        seed: Optional[Union[int, np.random.Generator]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.plan: Dict[Infostate, Action] = {}
        self.rng = np.random.default_rng(seed)

    def iterate(
        self,
        traversing_player: Optional[int] = None,
    ):
        traversing_player = self._cycle_updating_player(traversing_player)

        if self._verbose:
            print(
                "\nIteration",
                self._alternating_update_msg() if self.alternating else self.iteration,
            )
        root_reach_probabilities = (
            {player: 1.0 for player in [-1] + self.players}
            if self.simultaneous
            else None
        )
        # empty the previously sampled strategy
        self.plan.clear()

        self._traverse(
            self.root_state.clone(),
            root_reach_probabilities,
            traversing_player,
        )
        self._iteration += 1

    def _traverse(
        self,
        state: pyspiel.State,
        reach_prob: Optional[Dict[int, Probability]] = None,
        updating_player: Optional[int] = None,
    ):
        if state.is_terminal():
            return state.returns()

        if state.is_chance_node():
            outcomes_probs = state.chance_outcomes()
            outcome = sample_on_policy(
                values=[outcome[0] for outcome in outcomes_probs],
                policy=[outcome[1] for outcome in outcomes_probs],
                rng=self.rng,
            )
            return self._traverse(
                state.child(int(outcome)), reach_prob, updating_player
            )

        curr_player = state.current_player()
        infostate = state.information_state_string(curr_player)
        self._set_action_list(infostate, state)
        actions = self.action_list(infostate)
        regret_minimizer = self.regret_minimizer(infostate)
        player_policy = regret_minimizer.recommend(self.iteration)

        sampled_action = self._sample_action(infostate, player_policy)

        if self.simultaneous or curr_player == updating_player:

            if self.simultaneous:
                # increment the average policy for the player
                self._avg_policy_at(curr_player, infostate)[sampled_action] += 1
                # TODO: Simultaneous updating PURE CFR only works if we actually multiply the counterfactual reach
                #  probability to the regret increment. Therefore, I am mixing the regret update rule from
                #  chance-sampling and pure cfr: The state value is computed according to pure-car's sampled
                #  action-value, but the difference of each action value to the state value is then multiplied by
                #  the cf. reach probability as in chance-sampling. Why this ends up being a correct regret update
                #  is unclear, even more so because the policy update is exactly according to pure cfr, and not
                #  chance-sampling.
                prob_weight = counterfactual_reach_prob(reach_prob, curr_player)
            else:
                prob_weight = 1.0

            action_values = dict()
            for action in actions:
                child_reach_prob = deepcopy(reach_prob)
                if self.simultaneous:
                    child_reach_prob[action] *= player_policy[action]

                action_values[action] = self._traverse(
                    state.child(action),
                    child_reach_prob,
                    updating_player,
                )
            state_value = action_values[sampled_action]
            player_state_value = state_value[curr_player]
            regret_minimizer.observe_regret(
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

    def _avg_policy_at(self, current_player, infostate):
        if infostate not in (player_policy := self._avg_policy[current_player]):
            player_policy[infostate] = {
                action: 0.0 for action in self.action_list(infostate)
            }
        return player_policy[infostate]

    def _sample_action(
        self, infostate: Infostate, player_policy: Mapping[Action, Probability]
    ):
        if infostate not in self.plan:
            actions = self.action_list(infostate)
            self.plan[infostate] = sample_on_policy(
                values=actions,
                policy=[player_policy[action] for action in actions],
                rng=self.rng,
            )
        return self.plan[infostate]
