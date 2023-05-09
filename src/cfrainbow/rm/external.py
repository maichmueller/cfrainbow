import cmath
import math
from abc import ABC, abstractmethod
from typing import (
    Dict,
    Callable,
    Optional,
    Any,
    List,
)

from numba import njit

from .unbound import (
    hedge,
    regret_matching,
    regret_matching_plus,
    predictive_regret_matching,
    predictive_regret_matching_plus,
)
from ..spiel_types import *


class ExternalRegretMinimizer(ABC):
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

    @property
    def observes_regret(self):
        """Whether the regret minimizer expects to observe utilities or regrets."""
        return True

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
            self._recommendation_computed = True
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


def _parent_func(cls: type, obj: Any, func_name: str):
    if callable(
        _super_func := getattr(super(cls, obj), func_name, False)
    ) and not hasattr(_super_func, "__isabstractmethod__"):
        return _super_func
    else:
        return lambda *a, **k: None


class UtilityToRegretMixin:
    cumulative_quantity: Dict[Action, float]
    recommendation: Dict[Action, float]
    actions: List[Action]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.utility_storage = {a: 0.0 for a in self.actions}
        self._super__recommend = getattr(self, "_super__recommend", dict())
        self._super__recommend[UtilityToRegretMixin] = _parent_func(
            UtilityToRegretMixin, self, "_recommend"
        )

    @property
    def observes_regret(self):
        return False

    def _observe(
        self,
        iteration: int,
        regret_or_utility: Callable[[Action], float],
        *args,
        **kwargs,
    ):
        for action in self.cumulative_quantity.keys():
            self.utility_storage[action] += regret_or_utility(action)

    def _recommend(
        self,
        *args,
        **kwargs,
    ):
        expected_utility = sum(
            self.recommendation[action] * utility
            for action, utility in self.utility_storage.items()
        )
        for action, utility in self.utility_storage.items():
            self.cumulative_quantity[action] += utility - expected_utility
            self.utility_storage[action] = 0.0
        self._super__recommend[UtilityToRegretMixin](*args, **kwargs)


class RegretMatchingMixin:
    cumulative_quantity: Dict[Action, float]
    recommendation: Dict[Action, float]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._super__recommend = getattr(self, "_super__recommend", dict())
        self._super__recommend[RegretMatchingMixin] = _parent_func(
            RegretMatchingMixin, self, "_recommend"
        )

    def _regret_minimizer_impl(self, *args, **kwargs):
        return regret_matching(*args, **kwargs)

    def _predictive_regret_minimizer_impl(self, *args, **kwargs):
        return predictive_regret_matching(*args, **kwargs)

    def _recommend(
        self,
        *args,
        prediction: Optional[Mapping[Action, Value]] = None,
        **kwargs,
    ):
        if prediction is None:
            self._regret_minimizer_impl(self.recommendation, self.cumulative_quantity)
        else:
            self._predictive_regret_minimizer_impl(
                prediction, self.recommendation, self.cumulative_quantity
            )
        self._super__recommend[RegretMatchingMixin](
            *args, prediction=prediction, **kwargs
        )


class RegretMatchingPlusMixin(RegretMatchingMixin):
    def _regret_minimizer_impl(self, *args, **kwargs):
        return regret_matching_plus(*args, **kwargs)

    def _predictive_regret_minimizer_impl(self, *args, **kwargs):
        return predictive_regret_matching_plus(*args, **kwargs)


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


class RegretDiscounterMixin:
    cumulative_quantity: Dict[Action, float]
    recommendation: Dict[Action, float]
    actions: List[Action]
    _last_update_time: int

    def __init__(self, *args, alpha: float, beta: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self._super__recommend = getattr(self, "_super__recommend", dict())
        self._super__recommend[RegretDiscounterMixin] = _parent_func(
            RegretDiscounterMixin, self, "_recommend"
        )

    def _apply_weights(self):
        alpha, beta = weights(self._last_update_time + 1, self.alpha, self.beta)
        for action, regret in self.cumulative_quantity.items():
            self.cumulative_quantity[action] = regret * (alpha if regret > 0 else beta)

    def _recommend(
        self,
        *args,
        **kwargs,
    ):
        self._apply_weights()
        self._super__recommend[RegretDiscounterMixin](*args, **kwargs)


class AutoPredictiveMixin:
    """ """

    cumulative_quantity: Dict[Action, float]
    recommendation: Dict[Action, float]
    actions: List[Action]
    _last_update_time: int
    observes_regret: property

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_seen_quantity = {a: 0.0 for a in self.actions}

        self._super__recommend = getattr(self, "_super__recommend", dict())
        self._super__recommend[AutoPredictiveMixin] = _parent_func(
            AutoPredictiveMixin, self, "_recommend"
        )
        self._super__observe = getattr(self, "_super__observe", dict())
        self._super__observe[AutoPredictiveMixin] = _parent_func(
            AutoPredictiveMixin, self, "_observe"
        )

    def _recommend(
        self,
        *args,
        **kwargs,
    ):
        if not self.observes_regret:
            prediction_expectation = sum(
                self.recommendation[a] * last_action_quantity
                for a, last_action_quantity in self._last_seen_quantity.items()
            )
            prediction = {
                a: v / prediction_expectation
                for a, v in self._last_seen_quantity.items()
            }
        else:
            prediction = self._last_seen_quantity
        self._super__recommend[AutoPredictiveMixin](
            *args, prediction=prediction, **kwargs
        )
        # delete the current storage after the entire recommendation update ran
        self._last_seen_quantity = {a: 0.0 for a in self.actions}

    def _observe(
        self,
        iteration: int,
        regret_or_utility: Callable[[Action], float],
        *args,
        **kwargs,
    ):
        # first call the parent's _observe method
        self._super__observe[AutoPredictiveMixin](
            iteration,
            regret_or_utility,
            *args,
            **kwargs,
        )
        # then add the input to the last seen quantity storage
        for action in self.actions:
            self._last_seen_quantity[action] += regret_or_utility(action)


class RegretMatcher(RegretMatchingMixin, ExternalRegretMinimizer):
    pass


class RegretMatcherPlus(RegretMatchingPlusMixin, ExternalRegretMinimizer):
    pass


class RegretMatcherDiscounted(
    RegretDiscounterMixin, RegretMatchingMixin, ExternalRegretMinimizer
):
    pass


class RegretMatcherDiscountedPlus(
    RegretDiscounterMixin, RegretMatchingPlusMixin, ExternalRegretMinimizer
):
    pass


class AutoPredictiveRegretMatcher(
    AutoPredictiveMixin, RegretMatchingMixin, ExternalRegretMinimizer
):
    pass


class AutoPredictiveRegretMatcherPlus(
    AutoPredictiveMixin, RegretMatchingPlusMixin, ExternalRegretMinimizer
):
    pass


@njit
def anytime_hedge_rate(iteration: int, nr_actions: int, *args, **kwargs) -> float:
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
    def __init__(
        self,
        actions: Iterable[Action],
        *args,
        learning_rate: Callable[
            [int, int, Sequence, Mapping], float
        ] = anytime_hedge_rate,
        **kwargs,
    ):
        super().__init__(actions, *args, **kwargs)
        # function computing the learning rate for the given iteration and nr of actions to consider
        self._learning_rate: Callable[
            [int, int, Sequence, Mapping], float
        ] = learning_rate

    @property
    def observes_regret(self):
        return False

    def _recommend(self, iteration: Optional[int] = None, *args, **kwargs):
        hedge(
            self.recommendation,
            self.cumulative_quantity,
            lr=self._learning_rate(iteration, len(self), *args, **kwargs),
        )
