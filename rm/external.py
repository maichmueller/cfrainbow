import cmath
from abc import ABC, abstractmethod
from typing import (
    Dict,
    Callable,
    Iterable,
    Optional,
    MutableMapping,
)

from numba import njit

from spiel_types import *


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
    regret_map: Mapping[Action, float],
    positivized_regret_map: Optional[MutableMapping[Action, float]] = None,
):
    pos_regret_dict = (
        positivized_regret_map if positivized_regret_map is not None else dict()
    )
    pos_regret_sum = 0.0
    for action, regret in regret_map.items():
        pos_action_regret = max(0.0, regret)
        pos_regret_dict[action] = pos_action_regret
        pos_regret_sum += pos_action_regret
    normalize_or_uniformize_policy(policy, pos_regret_dict, pos_regret_sum)


def regret_matching_plus(
    policy: MutableMapping[Action, Probability],
    regret_map: MutableMapping[Action, float],
):
    regret_matching(policy, regret_map, regret_map)


def predictive_regret_matching(
    prediction: Mapping[Action, float],
    policy: MutableMapping[Action, Probability],
    regret_map: Mapping[Action, float],
):
    pos_regret_sum = 0.0
    avg_prediction: float = sum(
        [prediction[action] * action_prob for action, action_prob in policy.items()]
    )
    positivized_regret_map = dict()
    for action, regret in regret_map.items():
        pos_regret = max(0.0, regret + (prediction[action] - avg_prediction))
        pos_regret_sum += pos_regret
        positivized_regret_map[action] = pos_regret
    normalize_or_uniformize_policy(policy, positivized_regret_map, pos_regret_sum)


def predictive_regret_matching_plus(
    prediction: Mapping[Action, float],
    policy: MutableMapping[Action, Probability],
    regret_map: MutableMapping[Action, float],
):
    pos_regret_sum = 0.0
    avg_prediction = sum(
        [prediction[action] * action_prob for action, action_prob in policy.items()]
    )
    positivized_regret_map = dict()
    for action, regret in regret_map.items():
        pos_regret = max(0.0, regret)
        pos_regret_sum += pos_regret
        regret_map[action] = pos_regret
        positivized_regret_map[action] = max(
            0.0, pos_regret + prediction[action] - avg_prediction
        )

    normalize_or_uniformize_policy(policy, positivized_regret_map, pos_regret_sum)


class ExternalRegretMinimizer(ABC):
    def __init__(self, actions: Iterable[Action], *args, **kwargs):
        self._actions = list(actions)
        self.cumulative_regret: Dict[Action, float] = {a: 0.0 for a in self.actions}
        self.recommendation: Dict[Action, Probability] = {}
        self._recommendation_computed: bool = False

    @property
    def actions(self):
        return self._actions

    def regret(self, action: Action):
        return self.cumulative_regret[action]

    def reset(self):
        for action in self.actions:
            self.cumulative_regret[action] = 0.0
        self.recommendation.clear()
        self._recommendation_computed = False

    @abstractmethod
    def recommend(
        self, iteration: Optional[int] = None, *args, **kwargs
    ) -> Dict[Action, Probability]:
        raise NotImplementedError(
            f"method '{self.recommend.__name__}' is not implemented."
        )

    @abstractmethod
    def observe_regret(
        self, iteration: int, regret: Callable[[Action, Action], float], *args, **kwargs
    ):
        raise NotImplementedError(
            f"method '{self.observe_regret.__name__}' is not implemented."
        )

    @abstractmethod
    def observe_loss(
        self, iteration: int, loss: Callable[[Action], float], *args, **kwargs
    ):
        raise NotImplementedError(
            f"method '{self.observe_loss.__name__}' is not implemented."
        )


class RegretMatcher(ExternalRegretMinimizer):
    def __init__(self, actions: Iterable[Action]):
        actions = list(actions)
        uniform_prob = 1.0 / len(actions)
        super().__init__(actions)
        self.recommendation = {a: uniform_prob for a in actions}
        self._last_update_time: int = -1

    def reset(self):
        super().reset()
        self._last_update_time = -1

    def recommend(self, iteration: int = None, force: bool = False, *args, **kwargs):
        if force or (
            not self._recommendation_computed
            and self._last_update_time
            < iteration  # 2nd condition checks if the update iteration has completed
        ):
            self._prepare_recommendation(*args, **kwargs)
        return self.recommendation

    def observe_regret(
        self, iteration: int, regret: Callable[[Action], float], *args, **kwargs
    ):
        for action in self.cumulative_regret.keys():
            self.cumulative_regret[action] += regret(action)
        self._last_update_time = iteration
        self._recommendation_computed = False

    def _prepare_recommendation(self, *args, **kwargs):
        regret_matching(self.recommendation, self.cumulative_regret)
        self._recommendation_computed = True

    def observe_loss(
        self, iteration: int, loss: Callable[[Action], float], *args, **kwargs
    ):
        self.observe_regret(iteration, lambda a: loss(a) * self.recommendation[a])
        self._last_update_time = iteration
        self._recommendation_computed = False


class RegretMatcherPlus(RegretMatcher):
    def _prepare_recommendation(self):
        regret_matching_plus(self.recommendation, self.cumulative_regret)
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


class RegretMatcherDiscounted(RegretMatcher):
    def __init__(self, actions: Iterable[Action], alpha: float, beta: float):
        super().__init__(actions)
        self.alpha = alpha
        self.beta = beta

    def _apply_weights(self):
        alpha, beta = weights(self._last_update_time + 1, self.alpha, self.beta)
        for action, regret in self.cumulative_regret.items():
            self.cumulative_regret[action] = regret * (alpha if regret > 0 else beta)

    def _prepare_recommendation(self):
        self._apply_weights()
        super()._prepare_recommendation()


class RegretMatcherDiscountedPlus(RegretMatcherDiscounted):
    def _prepare_recommendation(self):
        self._apply_weights()
        regret_matching_plus(self.recommendation, self.cumulative_regret)
        self._recommendation_computed = True


class RegretMatcherPredictive(RegretMatcher):
    def _prepare_recommendation(self, prediction, *args, **kwargs):
        predictive_regret_matching(
            prediction, self.recommendation, self.cumulative_regret
        )
        self._recommendation_computed = True


class RegretMatcherPredictivePlus(RegretMatcher):
    def _prepare_recommendation(self, prediction, *args, **kwargs):
        predictive_regret_matching_plus(
            prediction, self.recommendation, self.cumulative_regret
        )
        self._recommendation_computed = True
