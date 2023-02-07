import cmath
import math
from abc import ABC, abstractmethod
from typing import (
    Dict,
    Callable,
    Optional,
    MutableMapping,
)

from numba import njit

from src.cfrainbow.spiel_types import *


def normalize_or_uniformize_policy(policy, pos_regret_dict, pos_regret_sum):
    if pos_regret_sum > 0.0:
        for action, pos_regret in pos_regret_dict.items():
            policy[action] = pos_regret / pos_regret_sum
    else:
        uniform_prob = 1 / len(policy)
        for action in policy.keys():
            policy[action] = uniform_prob


def regret_matching(
    policy: MutableMapping[Action, Probability],
    cumul_regret_map: Mapping[Action, float],
    positivized_regret_map: Optional[MutableMapping[Action, float]] = None,
):
    pos_regret_dict = (
        positivized_regret_map if positivized_regret_map is not None else dict()
    )
    pos_regret_sum = 0.0
    for action, regret in cumul_regret_map.items():
        pos_action_regret = max(0.0, regret)
        pos_regret_dict[action] = pos_action_regret
        pos_regret_sum += pos_action_regret
    normalize_or_uniformize_policy(policy, pos_regret_dict, pos_regret_sum)


def regret_matching_plus(
    policy: MutableMapping[Action, Probability],
    cumul_regret_map: MutableMapping[Action, float],
):
    regret_matching(policy, cumul_regret_map, cumul_regret_map)


def predictive_regret_matching(
    prediction: Mapping[Action, float],
    policy: MutableMapping[Action, Probability],
    cumul_regret_map: Mapping[Action, float],
):
    pos_regret_sum = 0.0
    avg_prediction: float = sum(
        [prediction[action] * action_prob for action, action_prob in policy.items()]
    )
    positivized_regret_map = dict()
    for action, regret in cumul_regret_map.items():
        pos_regret = max(0.0, regret + (prediction[action] - avg_prediction))
        pos_regret_sum += pos_regret
        positivized_regret_map[action] = pos_regret
    normalize_or_uniformize_policy(policy, positivized_regret_map, pos_regret_sum)


def predictive_regret_matching_plus(
    prediction: Mapping[Action, float],
    policy: MutableMapping[Action, Probability],
    cumul_regret_map: MutableMapping[Action, float],
):
    pos_regret_sum = 0.0
    avg_prediction = sum(
        [prediction[action] * action_prob for action, action_prob in policy.items()]
    )
    positivized_regret_map = dict()
    for action, regret in cumul_regret_map.items():
        pos_regret = max(0.0, regret)
        pos_regret_sum += pos_regret
        cumul_regret_map[action] = pos_regret
        positivized_regret_map[action] = max(
            0.0, pos_regret + prediction[action] - avg_prediction
        )
    normalize_or_uniformize_policy(policy, positivized_regret_map, pos_regret_sum)


def hedge(
    policy: MutableMapping[Action, Probability],
    cumul_utility_map: Mapping[Action, float],
    lr: float,
):
    sum_weights = 0.0
    for action, utility in cumul_utility_map.items():
        new_weight = math.exp(-lr * utility)
        sum_weights += new_weight
        policy[action] = new_weight
    for action in policy:
        policy[action] /= sum_weights


class ExternalRegretMinimizer(ABC):

    regret_mode: bool = True

    def __init__(self, actions: Iterable[Action], *args, **kwargs):
        self._actions = list(actions)
        self._recommendation_computed: bool = False
        self._n_actions = len(self._actions)
        self._last_update_time: int = -1

        # the initial recommendation always suggests a uniform play over all actions
        uniform_prob = 1.0 / len(self.actions)
        self.recommendation: Dict[Action, Probability] = {
            a: uniform_prob for a in actions
        }
        # the cumulative quantity is either...
        # 1. the cumulative regret (if regret based)
        # 2. the cumulative utility (else)
        # it is the storage of the quantity upon which the minimizer bases its recommendations.
        self.cumulative_quantity: Dict[Action, float] = {a: 0.0 for a in self.actions}

    def __len__(self):
        return self._n_actions


    @property
    def actions(self):
        return self._actions

    @property
    def last_update_time(self):
        return self._last_update_time

    def reset(self):
        for action in self.actions:
            self.cumulative_quantity[action] = 0.0
        self.recommendation.clear()
        self._recommendation_computed = False
        self._last_update_time: int = -1

    def recommend(self, iteration: int, *args, force: bool = False, **kwargs):
        if force or (
            not self._recommendation_computed
            and self._last_update_time
            < iteration  # 2nd condition checks if the update iteration has completed
        ):
            self._recommend(iteration, *args, **kwargs)
        return self.recommendation

    def observe(
        self,
        iteration: int,
        regret_or_utility: Callable[[Action], float],
        *args,
        **kwargs,
    ):
        self._observe(iteration, regret_or_utility, *args, **kwargs)
        self._last_update_time = iteration
        self._recommendation_computed = False

    def _observe(
        self,
        iteration: int,
        regret_or_utility: Callable[[Action], float],
        *args,
        **kwargs,
    ):
        for action in self.cumulative_quantity.keys():
            self.cumulative_quantity[action] += regret_or_utility(action)

    @abstractmethod
    def _recommend(
        self, iteration: Optional[int] = None, *args, **kwargs
    ) -> Dict[Action, Probability]:
        raise NotImplementedError(
            f"method '{self._recommend.__name__}' is not implemented."
        )


class RegretMatcher(ExternalRegretMinimizer):
    def _recommend(self, *args, **kwargs):
        regret_matching(self.recommendation, self.cumulative_quantity)
        self._recommendation_computed = True


class RegretMatcherPlus(ExternalRegretMinimizer):
    def _recommend(self, *args, **kwargs):
        regret_matching_plus(self.recommendation, self.cumulative_quantity)
        self._recommendation_computed = True


@njit
def weights(t: int, alpha: float, beta: float):
    if alpha < cmath.inf:
        alpha_weight = t**alpha
        alpha_weight /= alpha_weight + 1
    else:
        alpha_weight = 1
    if beta > -cmath.inf:
        beta_weight = t**beta
        beta_weight /= beta_weight + 1
    else:
        beta_weight = 0
    return alpha_weight, beta_weight


class RegretMatcherDiscounted(ExternalRegretMinimizer):
    def __init__(self, actions: Iterable[Action], alpha: float, beta: float):
        super().__init__(actions)
        self.alpha = alpha
        self.beta = beta

    def _apply_weights(self):
        alpha, beta = weights(self._last_update_time + 1, self.alpha, self.beta)
        for action, regret in self.cumulative_quantity.items():
            self.cumulative_quantity[action] = regret * (alpha if regret > 0 else beta)

    def _recommend(self, *args, **kwargs):
        self._apply_weights()
        regret_matching(self.recommendation, self.cumulative_quantity)
        self._recommendation_computed = True


class RegretMatcherDiscountedPlus(RegretMatcherDiscounted):
    def _recommend(self, *args, **kwargs):
        self._apply_weights()
        regret_matching_plus(self.recommendation, self.cumulative_quantity)
        self._recommendation_computed = True


class RegretMatcherPredictive(RegretMatcher):
    def _recommend(self, iteration, prediction, *args, **kwargs):
        predictive_regret_matching(
            prediction, self.recommendation, self.cumulative_quantity
        )
        self._recommendation_computed = True


class RegretMatcherPredictivePlus(RegretMatcher):
    def _recommend(self, iteration, prediction, *args, **kwargs):
        predictive_regret_matching_plus(
            prediction, self.recommendation, self.cumulative_quantity
        )
        self._recommendation_computed = True


@njit
def anytime_hedge_rate(iteration: int, nr_actions: int) -> float:
    """
    Learning rate of the hedge algorithm to achieve anytime O(sqrt(log(N) * T)) regret.

    Notes
    -----
    The name and regret bound stems from [1].

    [1] Mourtada, J. and Gaïffas, S., 2019.
        On the optimality of the Hedge algorithm in the stochastic regime.
        Journal of Machine Learning Research, 20, pp.1-28.

    Parameters
    ----------
    iteration: int
        the round counter for the hedge update.
    nr_actions: int
        how many actions are available and considered by the updating player.

    Returns
    -------
    float
        the learning rate for the anytime hedge update.
    """
    return math.sqrt(nr_actions * 100 / (iteration + 1))


class Hedge(ExternalRegretMinimizer):

    regret_mode = False

    def __init__(
        self,
        actions: Iterable[Action],
        *args,
        learning_rate: Callable[[int, int], float] = anytime_hedge_rate,
        **kwargs,
    ):
        super().__init__(actions, *args, **kwargs)
        # function computing the learning rate for the given iteration and nr of actions to consider
        self._learning_rate: Callable[[int, int], float] = learning_rate

    def _recommend(self, iteration: Optional[int] = None, *args, **kwargs):
        hedge(
            self.recommendation,
            self.cumulative_quantity,
            lr=self._learning_rate(iteration, len(self)),
        )
        self._recommendation_computed = True
