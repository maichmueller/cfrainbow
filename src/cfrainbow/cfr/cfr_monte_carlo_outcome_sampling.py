from collections import defaultdict
from copy import deepcopy, copy
from enum import Enum
from typing import Optional, Dict, Sequence

import pyspiel

from cfrainbow.spiel_types import Probability, Action, Player, Infostate
from cfrainbow.utils import (
    counterfactual_reach_prob,
    sample_on_policy,
)
from .cfr_base import CFRBase, iterate_logging


class OutcomeSamplingWeightingMode(Enum):
    lazy = 0
    optimistic = 1
    stochastic = 2


class OutcomeSamplingMCCFR(CFRBase):
    def __init__(
        self,
        *args,
        weighting_mode: OutcomeSamplingWeightingMode = OutcomeSamplingWeightingMode.stochastic,
        epsilon: float = 0.6,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.weighting_mode = OutcomeSamplingWeightingMode(weighting_mode)
        self.weight_storage: Dict[Infostate, Dict[Action, float]] = dict()
        self.epsilon = epsilon
        self.last_visit: defaultdict[Infostate, int] = defaultdict(int)

    def _weight(self, infostate, actions: Optional[Sequence[Action]] = None):
        if infostate not in self.weight_storage:
            assert actions is not None, (
                "Lazy weight storage has not been previously initialized. "
                "Call requires the valid action set to be passed as well in this case."
            )
            self.weight_storage[infostate] = {action: 0.0 for action in actions}
        return self.weight_storage[infostate]

    @iterate_logging
    def iterate(
        self,
        updating_player: Optional[Player] = None,
    ):
        value, tail_prob = self._traverse(
            self.root_state.clone(),
            reach_prob={player: 1.0 for player in [-1] + self.players},
            updating_player=self._cycle_updating_player(updating_player),
            sample_probability=1.0,
            weights=[0.0] * self.nr_players
            if self.weighting_mode == OutcomeSamplingWeightingMode.lazy
            else None,
        )
        self._iteration += 1
        return value

    def _traverse(
        self,
        state: pyspiel.State,
        reach_prob: Dict[Player, float],
        updating_player: Optional[Player] = None,
        sample_probability=1.0,
        weights: Optional[Dict[Player, float]] = None,
    ):
        self._nodes_touched += 1

        curr_player = state.current_player()
        if state.is_terminal():
            return [r / sample_probability for r in state.returns()], 1.0

        elif state.is_chance_node():
            sampled_action, action_index, sample_policy = sample_on_policy(
                *zip(*state.chance_outcomes()),
                rng=self.rng,
            )
            state.apply_action(int(sampled_action))
            sample_prob = sample_policy[action_index]
            reach_prob[curr_player] *= sample_prob
            return self._traverse(
                state,
                reach_prob,
                updating_player,
                sample_probability * sample_prob,
                weights=weights,
            )

        else:
            curr_player = state.current_player()
            infostate = state.information_state_string(curr_player)
            self._set_action_list(infostate, state)
            regret_minimizer = self.regret_minimizer(infostate)
            player_policy = regret_minimizer.recommend(self.iteration)

            (
                sampled_action,
                sampled_action_policy_prob,
                sampled_action_sample_prob,
            ) = self._sample_action(curr_player, updating_player, player_policy)

            child_reach_prob = self.child_reach_prob_map(
                reach_prob, curr_player, sampled_action_policy_prob
            )

            next_weights = copy(weights)
            if self.weighting_mode == OutcomeSamplingWeightingMode.lazy:
                next_weights[curr_player] = (
                    next_weights[curr_player] * sampled_action_policy_prob
                    + self._weight(infostate, self.action_list(infostate))[
                        sampled_action
                    ]
                )

            state.apply_action(sampled_action)
            action_value, tail_prob = self._traverse(
                state,
                child_reach_prob,
                updating_player,
                sample_probability * sampled_action_sample_prob,
                weights=next_weights,
            )

            def avg_policy_update_call():
                if player_reach_prob := reach_prob[curr_player] != 0.0:
                    self._update_average_policy(
                        curr_player,
                        infostate,
                        player_policy,
                        sampled_action,
                        player_reach_prob,
                        sample_probability,
                        weights,
                    )

            def regret_update_call():
                cf_value_weight = action_value[curr_player] * counterfactual_reach_prob(
                    reach_prob, curr_player
                )
                unsampled_action_regret = (
                    -cf_value_weight * tail_prob * sampled_action_policy_prob
                )

                def regret(action):
                    if action != sampled_action:
                        return unsampled_action_regret
                    else:
                        return (
                            cf_value_weight
                            * tail_prob
                            * (1 - sampled_action_policy_prob)
                        )

                regret_minimizer.observe(self.iteration, regret)

            if self.simultaneous:
                regret_update_call()
                avg_policy_update_call()
            else:
                if curr_player == updating_player:
                    regret_update_call()
                elif curr_player == self._peek_at_next_updating_player():
                    avg_policy_update_call()
            return action_value, tail_prob * sampled_action_policy_prob

    def _update_average_policy(
        self,
        curr_player,
        infostate,
        player_policy,
        sampled_action,
        player_reach_prob,
        sample_probability,
        weights,
    ):
        avg_policy = self._avg_policy_at(curr_player, infostate)
        if self.weighting_mode == OutcomeSamplingWeightingMode.optimistic:
            last_visit_difference = self.iteration + 1 - self.last_visit[infostate]
            self.last_visit[infostate] = self.iteration
            for action, policy_prob in player_policy.items():
                avg_policy[action] += (
                    player_reach_prob * policy_prob * last_visit_difference
                )
        elif self.weighting_mode == OutcomeSamplingWeightingMode.stochastic:
            for action, policy_prob in player_policy.items():
                avg_policy[action] += (
                    player_reach_prob * policy_prob / sample_probability
                )
        else:
            # lazy weighting updates
            stored_weights = self._weight(infostate)
            for action, policy_prob in player_policy.items():
                policy_incr = (weights[curr_player] + player_reach_prob) * policy_prob
                avg_policy[action] += policy_incr
                if action != sampled_action:
                    stored_weights[action] += policy_incr
                else:
                    stored_weights[action] = 0.0

    def _sample_action(
        self,
        current_player: Player,
        updating_player: Optional[Player],
        policy: Dict[Action, Probability],
    ):
        actions = list(policy.keys())
        sampled_action, sample_index, sample_policy = sample_on_policy(
            actions,
            [policy[action] for action in actions],
            self.rng,
            epsilon=(
                self.epsilon * (self.simultaneous or current_player == updating_player)
            ),
        )
        return sampled_action, policy[sampled_action], sample_policy[sample_index]
