import cmath
import itertools
from collections import defaultdict
from typing import Sequence, Dict, Mapping, Tuple

from cfrainbow.spiel_types import (
    NormalFormPlan,
    NormalFormStrategy,
    NormalFormStrategySpace,
    Probability,
)


def cce_deviation_incentive(
    joint_distribution: NormalFormStrategy,
    strategy_spaces: Sequence[NormalFormStrategySpace],
    payoff: Mapping[Tuple[NormalFormPlan], Sequence[float]],
):
    r"""Computes the lowest coarse correlated equilibrium deviation incentive of a joint normal-form distribution.

    Notes
    -----
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
    r"""Computes the lowest correlated equilibrium deviation incentive of a joint normal-form distribution.

    Notes
    -----
    A correlated equilibrium :math:`\mu` is a probability distribution over the joint normal-form strategy space
    of all players that fulfills the following criteria:
    Let :math:`S_i` be the strategy space of player :math:`i` and :math:`S_{-i}` be the strategy space of :math:`i`'s
    opponents :math:`S_1 \times \dots S_{i-1} \times S_{i+1} \times \dots \times S_n`, a probability distribution
    :math:`\mu` over the joint strategy space :math:`S_1 \times \dots \times S_n` is an :math:`\epsilon` - CE for some
    :math:`\epsilon > 0` iff for each player :math:`i` and :math:`\forall s_i, z_i \in S_i` with
    :math:`\mu_i(s_i) = \sum_{s_{-i} \in S_{-i}} \mu(s_i, s_{-i})` being the marginal likelihood of s_i holds:

    .. math::
            \sum_{s_{-i} \in S_{-i}} \frac{\mu(s_i, s_{-i})}{\mu_i(s_i)} (u(z_i, s_{-i}) - u(s_i, s_{-i})) \leq \epsilon


    Parameters
    ----------
    joint_distribution: NormalFormStrategy
        the distribution whose distance to a CE is to be evaluated.
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
