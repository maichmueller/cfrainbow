from collections import defaultdict
from copy import deepcopy
from enum import Enum
from typing import Optional, Dict

import numpy as np
import pyspiel

from cfrainbow.spiel_types import Probability, Action
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
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.weighting_mode = OutcomeSamplingWeightingMode(weighting_mode)
        self.weight_storage: Dict[str, Dict[int, float]] = dict()
        self.epsilon = epsilon
        self.last_visit: defaultdict[str, int] = defaultdict(int)

    def _weight(self, infostate):
        if infostate not in self.weight_storage:
            self.weight_storage[infostate] = defaultdict(float)
        return self.weight_storage[infostate]

    @iterate_logging
    def iterate(
        self,
        updating_player: Optional[int] = None,
    ):
        value, tail_prob = self._traverse(
            deepcopy(self.root_state),
            reach_prob={player: 1.0 for player in [-1] + self.players},
            updating_player=self._cycle_updating_player(updating_player),
            sample_probability=1.0,
            weights={player: 0.0 for player in self.players}
            if self.weighting_mode == OutcomeSamplingWeightingMode.lazy
            else None,
        )
        self._iteration += 1
        return value

    def _traverse(
        self,
        state: pyspiel.State,
        reach_prob: dict[int, float],
        updating_player: Optional[int] = None,
        sample_probability=1.0,
        weights: Optional[dict[int, float]] = None,
    ):
        self._nodes_touched += 1

        curr_player = state.current_player()
        if state.is_terminal():
            reward, sample_prob = (
                np.asarray(state.returns()) / sample_probability,
                1.0,
            )
            return reward, sample_prob

        if state.is_chance_node():
            chance_policy = state.chance_outcomes()
            sampled_action, action_index, sample_policy = sample_on_policy(
                values=[outcome for outcome, prob in chance_policy],
                policy=[prob for outcome, prob in chance_policy],
                rng=self.rng,
            )
            sample_prob = sample_policy[action_index]
            reach_prob[curr_player] *= sample_prob
            state.apply_action(int(sampled_action))
            return self._traverse(
                state,
                reach_prob,
                updating_player,
                sample_probability * sample_prob,
                weights=weights,
            )

        curr_player = state.current_player()
        infostate = state.information_state_string(curr_player)
        self._set_action_list(infostate, state)
        regret_minimizer = self.regret_minimizer(infostate)
        player_policy = regret_minimizer.recommend(self.iteration)

        (
            sampled_action,
            sampled_action_prob,
            sampled_action_sample_prob,
        ) = self._sample_action(curr_player, updating_player, player_policy)

        child_reach_prob = deepcopy(reach_prob)
        child_reach_prob[curr_player] *= sampled_action_prob
        next_weights = deepcopy(weights)
        if self.weighting_mode == OutcomeSamplingWeightingMode.lazy:
            next_weights[curr_player] = (
                next_weights[curr_player] * sampled_action_prob
                + self._weight(infostate)[sampled_action]
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
            self._update_average_policy(
                curr_player,
                infostate,
                player_policy,
                sampled_action,
                reach_prob,
                sample_probability,
                weights,
            )

        if self.simultaneous or updating_player == curr_player:
            cf_value_weight = action_value[curr_player] * counterfactual_reach_prob(
                reach_prob, curr_player
            )
            regret_minimizer.observe(
                self.iteration,
                lambda action: (
                    cf_value_weight
                    * tail_prob
                    * (
                        (action == sampled_action)
                        * (1.0 - player_policy[sampled_action])
                        - (not (action == sampled_action))
                        * player_policy[sampled_action]
                    )
                ),
            )
            if self.simultaneous:
                avg_policy_update_call()
        else:
            avg_policy_update_call()
        return action_value, tail_prob * sampled_action_prob

    def _update_average_policy(
        self,
        curr_player,
        infostate,
        player_policy,
        sampled_action,
        reach_prob,
        sample_probability,
        weights,
    ):
        avg_policy = self._avg_policy_at(curr_player, infostate)
        if self.weighting_mode == OutcomeSamplingWeightingMode.optimistic:
            last_visit_difference = self.iteration + 1 - self.last_visit[infostate]
            self.last_visit[infostate] = self.iteration
            for action, policy_prob in player_policy.items():
                avg_policy[action] += (
                    reach_prob[curr_player] * policy_prob * last_visit_difference
                )
        elif self.weighting_mode == OutcomeSamplingWeightingMode.stochastic:
            for action, policy_prob in player_policy.items():
                avg_policy[action] += (
                    reach_prob[curr_player] * policy_prob / sample_probability
                )
        else:
            # lazy weighting updates
            for action, policy_prob in player_policy.items():
                policy_incr = (
                    weights[curr_player] + reach_prob[curr_player]
                ) * policy_prob
                avg_policy[action] += policy_incr
                weights[curr_player] = (weights[curr_player] + policy_incr) * (
                    action != sampled_action
                )

    def _sample_action(
        self,
        current_player: int,
        updating_player: int,
        policy: Dict[Action, Probability],
    ):
        actions = list(policy.keys())
        sampled_action, sample_index, sample_policy = sample_on_policy(
            actions,
            [policy[action] for action in actions],
            self.rng,
            epsilon=self.epsilon * (current_player == updating_player),
        )
        return sampled_action, policy[sampled_action], sample_policy[sample_index]
