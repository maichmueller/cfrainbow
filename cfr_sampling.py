from collections import defaultdict, deque
from copy import deepcopy
from typing import Optional, Dict, Sequence, Union, Type, MutableMapping, Mapping

from cfr_base import StochasticCFRBase
from cfr_pure import PureCFR
from rm import regret_matching, ExternalRegretMinimizer
import pyspiel
import numpy as np

from utils import sample_on_policy, counterfactual_reach_prob
from type_aliases import Action, Infostate, Probability, Regret, Value


class SamplingCFR(PureCFR):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        kwargs["alternating"] = False
        super().__init__(*args, **kwargs)
        self.plan: Dict[Infostate, Action] = dict()
        # the storage of all played strategy profiles in the iterations.
        # 'sum of play' since it is pre division by 'T'.
        self.empirical_sum_of_play: Dict[tuple[tuple[Infostate, Action], ...], int] = defaultdict(int)

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
        # empty the previously sampled strategy profile
        self.plan = dict()
        self._traverse(
            self.root_state.clone(),
            root_reach_probabilities,
            traversing_player,
        )
        self.empirical_sum_of_play[tuple(self.plan.items())] += 1
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
            return self._traverse_chance_node(state, reach_prob, updating_player)

        curr_player = state.current_player()
        infostate = state.information_state_string(curr_player)
        self._set_action_list(infostate, state)
        actions = self.action_list(infostate)
        regret_minimizer = self.regret_minimizer(infostate)
        player_policy = regret_minimizer.recommend(self.iteration)

        sampled_action = self._sample_action(infostate, player_policy)

        # increment the average policy for the player
        self._avg_policy_at(curr_player, infostate)[sampled_action] += 1

        action_values = dict()
        for action in actions:
            child_reach_prob = deepcopy(reach_prob)
            child_reach_prob[action] *= action == sampled_action

            action_values[action] = self._traverse(
                state.child(action),
                child_reach_prob,
                updating_player,
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
        state_value = np.zeros(self.nr_players)
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
            state_value += is_sampled_outcome * np.asarray(action_value)
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
            self.plan[infostate], _, _ = sample_on_policy(
                values=actions,
                policy=[player_policy[action] for action in actions],
                rng=self.rng,
            )
        return self.plan[infostate]
