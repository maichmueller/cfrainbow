import cmath
import itertools
from collections import namedtuple, defaultdict
from copy import copy
from functools import singledispatch
from typing import Sequence, Dict, Mapping, Tuple, Set, Optional

import numpy as np
import pyspiel

from type_aliases import (
    Infostate,
    Action,
    NormalFormPlan,
    NormalFormStrategy,
    NormalFormStrategySpace,
    Probability,
)
from utils import all_states_gen, infostates_gen


class InformedActionList:
    informed_tuple = namedtuple("informed_tuple", "action infostate depth")

    def __init__(self, infostate: Infostate, actions: Sequence[Action], depth: int):
        self.infostate: Infostate = infostate
        self.actions: Sequence[Action] = actions
        self.depth: int = depth

    def __iter__(self):
        return iter(
            self.informed_tuple(action, self.infostate, self.depth)
            for action in self.actions
        )

    def __repr__(self):
        return f"{self.infostate}, {self.actions}, {self.depth}"


def normal_form_strategy_space(
    game: pyspiel.Game, *players: int
) -> Dict[int, Set[NormalFormPlan]]:
    if not players:
        players = list(range(game.num_players()))
    action_spaces = {p: [] for p in players}
    strategy_spaces = {}
    seen_infostates = set()
    for state, depth in all_states_gen(root=game.new_initial_state()):
        player = state.current_player()
        if player in players:
            infostate = state.information_state_string(player)
            if infostate not in seen_infostates:
                action_spaces[player].append(
                    InformedActionList(infostate, state.legal_actions(), depth)
                )
                seen_infostates.add(infostate)

    for player, action_space in action_spaces.items():
        strategy_spaces[player] = set(
            tuple(
                (sorted_plan.infostate, sorted_plan.action)
                for sorted_plan in sorted(plan, key=lambda x: x.depth)
            )
            for plan in itertools.product(
                *sorted(
                    action_space,
                    key=lambda x: x.depth * 1e8 + sum(ord(c) for c in x.infostate),
                )
            )
        )
    return strategy_spaces


def normal_form_expected_payoff(game: pyspiel.Game, *joint_plan: NormalFormPlan):
    joint_plan_dict = {
        infostate: action for infostate, action in itertools.chain(*joint_plan)
    }
    root = game.new_initial_state()
    stack = [(root, 1.0)]
    expected_payoff = np.zeros(root.num_players())
    while stack:
        s, chance = stack.pop()
        if s.is_chance_node():
            for outcome, prob in s.chance_outcomes():
                stack.append((s.child(outcome), chance * prob))
        elif s.is_terminal():
            expected_payoff += chance * np.asarray(s.returns())
        else:
            infostate = s.information_state_string(s.current_player())
            if infostate not in joint_plan_dict:
                # the infostate not being part of the plan means this is a reduced plan and the current point is
                # unreachable by the player due to previous decisions. The expected outcome of this is hence simply 0,
                # since it is never reached.
                continue
            s.apply_action(joint_plan_dict[infostate])
            stack.append((s, chance))
    return expected_payoff


def normal_form_expected_payoff_table(
    game: pyspiel.Game, strategy_spaces: Sequence[NormalFormStrategySpace]
):
    payoffs: Dict[Tuple[NormalFormPlan], Sequence[float]] = dict()
    for joint_profile in itertools.product(*strategy_spaces):
        payoffs[joint_profile] = normal_form_expected_payoff(game, *joint_profile)
    return payoffs


def cce_deviation_incentive(
    joint_distribution: NormalFormStrategy,
    strategy_spaces: Sequence[NormalFormStrategySpace],
    payoff: Mapping[Tuple[NormalFormPlan], Sequence[float]],
):
    r"""
    Computes the epsilon-coarse correlated equilibrium deviation incentive for a joint normal-form distribution.

    A coarse correlated equilibrium :math:`\mu` is a probability distribution over the joint normal-form strategy space
    of all players that fulfills the following criteria:
    Let :math:`S_i` be the strategy space of player :math:`i` and :math:`S_{-i}` be the strategy space of :math:`i`'s
    opponents :math:`S_1 \times \dots S_{i-1} \times S_{i+1} \times \dots \times S_n`, a probability distribution
    :math:`\mu` over the joint strategy space :math:`S_1 \times \dots \times S_n` is an :math:`\epsilon` - CCE for some
    :math:`\epsilon > 0` iff for each player :math:`i` and :math:`\forall z_i \in S_i` holds:

    .. math::
            \sum_{s_i \in S_i} \sum_{s_{-i} \in S_{-i}} \mu(s_i, s_{-i}) (u(z_i, s_{-i}) - u(s_i, s_{-i})) \leq \epsilon


    Parameters
    ----------
    joint_distribution: NormalFormStrategy
        the distribution whose distance to a CCE is to be evaluated.
    strategy_spaces: Sequence[NormalFormStrategySpace]
        the sequence of players' strategy spaces. Element i is the ith player's strategy space.
    payoff: Mapping[Tuple[NormalFormPlan], Sequence[float]]
        the map of normal-form plan profiles to the vector of player payoffs. Payoff index i is ith player's payoff.

    Returns
    -------
    float
        the maximum deviation incentive for any player and any of their strategies.
    """
    strategy_spaces = list(strategy_spaces)
    deviation_incentive = -cmath.inf
    for player, strategy_space in enumerate(strategy_spaces):

        strategy_space = list(strategy_space)
        opponent_strategy_space = strategy_spaces.copy()
        opponent_strategy_space.pop(player)

        strategy_incentive: Dict[NormalFormPlan, float] = defaultdict(float)

        for joint_opp_plan in itertools.product(*opponent_strategy_space):
            for i, plan in enumerate(strategy_space):
                joint_plan = joint_opp_plan[:player] + (plan,) + joint_opp_plan[player:]
                joint_plan_prob = joint_distribution[joint_plan]
                joint_plan_payoff = payoff[joint_plan][player]

                for other_plan in strategy_space[i + 1 :]:
                    other_joint_plan = (
                        joint_opp_plan[:player]
                        + (other_plan,)
                        + joint_opp_plan[player:]
                    )
                    other_joint_plan_prob = joint_distribution[other_joint_plan]
                    other_joint_plan_payoff = payoff[other_joint_plan][player]

                    # add mu(s_i, s_{-i}) * (u[z_i, s_{-i}] - u[s_i, s_{-i}]),  s_i = plan, z_i = other_plan
                    strategy_incentive[plan] += joint_plan_prob * (
                        other_joint_plan_payoff - joint_plan_payoff
                    )
                    # and mu(z_i, s_{-i}) * (u[s_i, s_{-i}] - u[z_i, s_{-i}]),  s_i = plan, z_i = other_plan
                    strategy_incentive[other_plan] += other_joint_plan_prob * (
                        joint_plan_payoff - other_joint_plan_payoff
                    )
                    # to the respective deviation expectation
        player_deviation = max(strategy_incentive.values())
        deviation_incentive = max(deviation_incentive, player_deviation)
    return deviation_incentive


def ce_deviation_incentive(
    joint_distribution: NormalFormStrategy,
    strategy_spaces: Sequence[NormalFormStrategySpace],
    payoff: Mapping[Tuple[NormalFormPlan], Sequence[float]],
):
    strategy_spaces = list(strategy_spaces)
    deviation_incentive = -cmath.inf
    for player, strategy_space in enumerate(strategy_spaces):

        strategy_space = list(strategy_space)
        opponent_strategy_space = strategy_spaces.copy()
        opponent_strategy_space.pop(player)

        marginal_dist: Dict[NormalFormPlan, Probability] = defaultdict(float)
        exp_dev_sum: Dict[Tuple[NormalFormPlan, NormalFormPlan], float] = defaultdict(
            float
        )

        for joint_opp_plan in itertools.product(*opponent_strategy_space):
            for i, plan in enumerate(strategy_space):
                joint_plan = joint_opp_plan[:player] + (plan,) + joint_opp_plan[player:]
                joint_plan_prob = joint_distribution[joint_plan]
                joint_plan_payoff = payoff[joint_plan][player]

                marginal_dist[plan] += joint_plan_prob

                for other_plan in strategy_space[i + 1 :]:
                    other_joint_plan = (
                        joint_opp_plan[:player]
                        + (other_plan,)
                        + joint_opp_plan[player:]
                    )
                    other_joint_plan_prob = joint_distribution[other_joint_plan]
                    other_joint_plan_payoff = payoff[other_joint_plan][player]

                    exp_dev_sum[(plan, other_plan)] += joint_plan_prob * (
                        other_joint_plan_payoff - joint_plan_payoff
                    )
                    exp_dev_sum[(other_plan, plan)] += other_joint_plan_prob * (
                        joint_plan_payoff - other_joint_plan_payoff
                    )
        player_deviation = -cmath.inf
        for i, s in enumerate(strategy_space):
            for z in strategy_space[i + 1 :]:
                player_deviation = max(
                    player_deviation,
                    exp_dev_sum[(s, z)] / marginal_dist[s],
                    exp_dev_sum[(z, s)] / marginal_dist[z],
                )
        deviation_incentive = max(deviation_incentive, player_deviation)
    return deviation_incentive