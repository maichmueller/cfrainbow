from __future__ import annotations
import itertools
from collections import defaultdict
from copy import deepcopy
from typing import Optional, Dict, Mapping, Sequence, Union

import numpy as np
import pyspiel
from click import Tuple

from spiel_types import Action, Infostate, Probability, NormalFormPlan
from utils import sample_on_policy, counterfactual_reach_prob
from .cfr_base import iterate_log_print
from .cfr_pure import PureCFR


class JointNormalFormPlan:
    def __init__(self, plans: Sequence[NormalFormPlan]):
        self.plans: tuple[NormalFormPlan] = tuple(plans)
        self.hash = hash(tuple(elem for elem in itertools.chain(self.plans)))

    def __contains__(self, item):
        return any(item in plan for plan in self.plans)

    def __hash__(self):
        return self.hash

    def __eq__(self, other: Union[Sequence[NormalFormPlan], JointNormalFormPlan]):
        if isinstance(other, JointNormalFormPlan):
            return self.plans == other.plans
        return all(other_plan == plan for other_plan, plan in zip(other, self.plans))

    def __repr__(self):
        return repr(self.plans)


class SamplingCFR(PureCFR):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        kwargs["alternating"] = False
        super().__init__(*args, **kwargs)
        # the storage for each player's sampled strategy during the iteration
        self.plan: tuple[Dict[Infostate, Action], ...] = tuple(
            dict() for _ in self.players
        )
        # the storage of all played strategy profiles in the iterations.
        # 'sum of play' since it is prior to division by the iteration count 'T'.
        self.empirical_sum_of_play: Dict[JointNormalFormPlan, int] = defaultdict(int)

    @iterate_log_print
    def iterate(
        self,
        traversing_player: Optional[int] = None,
    ):
        # empty the previously sampled strategy profile
        for plan in self.plan:
            plan.clear()

        self._traverse(
            self.root_state.clone(),
            reach_prob={player: 1.0 for player in [-1] + self.players},
            traversing_player=self._cycle_updating_player(traversing_player),
        )
        self.empirical_sum_of_play[
            JointNormalFormPlan(
                tuple(tuple(plan.items()) for plan in itertools.chain(self.plan))
            )
        ] += 1
        self._iteration += 1

    def _traverse(
        self,
        state: pyspiel.State,
        reach_prob: Optional[Dict[int, Probability]] = None,
        traversing_player: Optional[int] = None,
    ):
        if state.is_terminal():
            return state.returns()

        if state.is_chance_node():
            return self._traverse_chance_node(state, reach_prob, traversing_player)

        curr_player = state.current_player()
        infostate = state.information_state_string(curr_player)
        self._set_action_list(infostate, state)
        actions = self.action_list(infostate)
        regret_minimizer = self.regret_minimizer(infostate)
        player_policy = regret_minimizer.recommend(self.iteration)

        sampled_action = self._sample_action(
            infostate, player_policy, sampling_player=curr_player
        )

        # increment the average policy for the player
        self._avg_policy_at(curr_player, infostate)[sampled_action] += 1

        action_values = self._action_value_map(infostate)
        for action in actions:
            child_reach_prob = deepcopy(reach_prob)
            child_reach_prob[action] *= action == sampled_action

            action_values[action] = self._traverse(
                state.child(action),
                child_reach_prob,
                traversing_player,
            )
        state_value = action_values[sampled_action]
        player_state_value = state_value[curr_player]

        cf_reach_prob = counterfactual_reach_prob(reach_prob, curr_player)
        regret_minimizer.observe_regret(
            self.iteration,
            lambda a: (
                cf_reach_prob * (action_values[a][curr_player] - player_state_value)
            ),
        )
        return state_value

    def _traverse_chance_node(self, state, reach_prob, updating_player):
        state_value = [0.0] * self.nr_players
        outcomes_probs = state.chance_outcomes()
        sampled_outcome, _, _ = sample_on_policy(
            values=[outcome[0] for outcome in outcomes_probs],
            policy=[outcome[1] for outcome in outcomes_probs],
            rng=self.rng,
        )
        for outcome, _ in outcomes_probs:
            next_state = state.child(outcome)

            is_sampled_outcome = outcome == sampled_outcome

            child_reach_prob = deepcopy(reach_prob)
            child_reach_prob[state.current_player()] *= is_sampled_outcome

            action_value = self._traverse(next_state, child_reach_prob, updating_player)
            for p in self.players:
                state_value[p] += is_sampled_outcome * action_value[p]
        return state_value

    def _sample_action(
        self,
        infostate: Infostate,
        player_policy: Mapping[Action, Probability],
        *args,
        **kwargs,
    ):
        if infostate not in (plan := self.plan[kwargs["sampling_player"]]):
            actions = self.action_list(infostate)
            plan[infostate], _, _ = sample_on_policy(
                values=actions,
                policy=[player_policy[action] for action in actions],
                rng=self.rng,
            )
        return plan[infostate]
