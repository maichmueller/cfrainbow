import cmath
from abc import ABC, abstractmethod
from typing import (
    Dict,
    List,
    Callable,
    Iterable,
    Optional,
    MutableMapping,
    Mapping,
    Union,
)
from type_aliases import *


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
    if pos_regret_sum > 0.0:
        for action, pos_regret in pos_regret_dict.items():
            policy[action] = pos_regret / pos_regret_sum
    else:
        uniform_prob = 1 / len(policy)
        for action in policy.keys():
            policy[action] = uniform_prob


def regret_matching_plus(
    policy: MutableMapping[Action, Probability],
    regret_map: MutableMapping[Action, float],
):
    regret_matching(policy, regret_map, regret_map)


def predictive_regret_matching(
    prediction,
    policy: MutableMapping[Action, Probability],
    regret_map: Mapping[Action, float],
    positivized_regret_map: Optional[MutableMapping[Action, float]] = None,
):
    pos_regret_dict = dict()
    pos_regret_sum = 0.0
    avg_prediction = sum(
        [prediction[action] * action_prob for action, action_prob in policy.items()]
    )
    for action, regret in regret_map.items():
        pos_action_regret = max(0, regret + (prediction[action] - avg_prediction))
        pos_regret_dict[action] = pos_action_regret
        pos_regret_sum += pos_action_regret
    if pos_regret_sum > 0.0:
        for action, pos_regret in pos_regret_dict.items():
            policy[action] = pos_regret / pos_regret_sum
    else:
        uniform_prob = 1 / len(pos_regret_dict)
        for action in policy.keys():
            policy[action] = uniform_prob


def predictive_regret_matching_plus(prediction, policy, regret_dict):
    pos_regret_sum = 0.0
    avg_prediction = sum(
        [prediction[action] * action_prob for action, action_prob in policy.items()]
    )
    for action, regret in regret_dict.items():
        pos_regret = max(0, regret)
        regret_dict[action] = pos_regret
        pos_action_regret = max(0, pos_regret + prediction[action] - avg_prediction)
        pos_regret_sum += pos_action_regret
    if pos_regret_sum > 0.0:
        for action, pos_regret in regret_dict.items():
            policy[action] = pos_regret / pos_regret_sum
    else:
        uniform_prob = 1 / len(regret_dict)
        for action in policy.keys():
            policy[action] = uniform_prob


#
# dict_ty = types.DictType(types.int64, types.unicode_type)
#
# specBase = [('cumulative_regret', NbDict.empty()),]
class ExternalRegretMinimizer(ABC):
    def __init__(self, actions: Iterable[Action], *args, **kwargs):
        self.cumulative_regret = {a: 0.0 for a in actions}
        self.recommendation: Dict[Action, Probability] = {}
        self._recommendation_computed: bool = False

    @abstractmethod
    def recommend(self, iteration: Optional[int] = None):
        raise NotImplementedError("'recommend' is not implemented.")

    @abstractmethod
    def observe_regret(
        self, iteration: int, loss: Callable[[Action], float], *args, **kwargs
    ):
        raise NotImplementedError("'observe_loss' is not implemented.")


class RegretMatcher(ExternalRegretMinimizer):
    def __init__(self, actions: Iterable[Action]):
        actions = list(actions)
        uniform_prob = 1.0 / len(actions)
        super().__init__(actions)
        self.recommendation = {a: uniform_prob for a in actions}
        self._last_update_time: int = 0

    def recommend(self, iteration: Optional[int] = None, force: bool = False):
        if force or (
            not self._recommendation_computed
            and self._last_update_time
            < iteration  # 2nd condition checks if the update iteration has completed
        ):
            self._ready_recommendation()
        return self.recommendation

    def observe_regret(
        self, iteration: int, loss: Callable[[Action], float], *args, **kwargs
    ):
        for action in self.cumulative_regret.keys():
            self.cumulative_regret[action] += loss(action)
        self._last_update_time = iteration
        self._recommendation_computed = False

    def _ready_recommendation(self):
        regret_matching(self.recommendation, self.cumulative_regret)
        self._recommendation_computed = True


class RegretMatcherPlus(RegretMatcher):
    def _ready_recommendation(self):
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

    def _ready_recommendation(self):
        self._apply_weights()
        super()._ready_recommendation()


class RegretMatcherDiscountedPlus(RegretMatcherDiscounted):
    def _ready_recommendation(self):
        self._apply_weights()
        regret_matching_plus(self.recommendation, self.cumulative_regret)
        self._recommendation_computed = True
