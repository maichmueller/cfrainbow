from abc import ABC, abstractmethod
from typing import (
    Dict,
    Callable,
    Iterable,
    Optional,
    Type,
)

import numpy as np

from type_aliases import *
from utils import slice_kwargs
from .external import ExternalRegretMinimizer


class InternalRegretMinimizer(ABC):
    def __init__(self, actions: Iterable[Action], *args, **kwargs):
        self.recommendation: Dict[Action, Probability] = {}
        self._actions = list(actions)
        self._recommendation_computed: bool = False

    @property
    def actions(self):
        return self._actions

    def reset(self):
        self.recommendation.clear()
        self._recommendation_computed = False

    @abstractmethod
    def regret(self, action_from: Action, action_to: Action):
        raise NotImplementedError(
            f"method '{self.regret.__name__}' is not implemented."
        )

    @abstractmethod
    def recommend(self, iteration: Optional[int] = None) -> Dict[Action, Probability]:
        raise NotImplementedError(
            f"method '{self.recommend.__name__}' is not implemented."
        )

    @abstractmethod
    def observe_regret(
        self, iteration: int, regret: Callable[[Action], float], *args, **kwargs
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


class InternalFromExternalRegretMinimizer(InternalRegretMinimizer):
    def __init__(
        self,
        actions: Iterable[Action],
        external_regret_minimizer_type: Type[ExternalRegretMinimizer],
        *args,
        **kwargs,
    ):
        super().__init__(actions, *args, **kwargs)
        self._regret_minimizer_kwargs = slice_kwargs(
            kwargs, external_regret_minimizer_type.__init__
        )

        self.external_minimizer = {
            a: external_regret_minimizer_type(
                self.actions, **self._regret_minimizer_kwargs
            )
            for a in self.actions
        }

        self._last_update_time = -1

    def reset(self):
        for minimizer in self.external_minimizer.values():
            minimizer.reset()
        self.recommendation.clear()
        self._recommendation_computed = False

    def regret(self, action_from: Action, action_to: Action):
        return self.external_minimizer[action_from].regret(action_to)

    def recommend(self, iteration: int = None, force: bool = False):
        if force or (
            not self._recommendation_computed
            and self._last_update_time
            < iteration  # 2nd condition checks if the update iteration has completed
        ):
            self._ready_recommendation(iteration, force)
        return self.recommendation

    def observe_regret(
        self, iteration: int, regret: Callable[[Action, Action], float], *args, **kwargs
    ):
        for assigned_action, erm in self.external_minimizer.items():
            erm.observe_regret(iteration, lambda a: regret(assigned_action, a))
        self._last_update_time = iteration
        self._recommendation_computed = False

    def observe_loss(
        self, iteration: int, loss: Callable[[Action], float], *args, **kwargs
    ):
        for assigned_action, erm in self.external_minimizer.items():
            erm.observe_regret(
                iteration,
                lambda a: loss(a)
                * self.recommendation[assigned_action]
                * erm.recommendation[a],
            )
        self._last_update_time = iteration
        self._recommendation_computed = False

    def _ready_recommendation(self, iteration: int = None, force: bool = False):
        n_actions = len(self.actions)
        # build the recommendations matrix
        recommendations = np.empty(shape=(n_actions, n_actions), dtype=float)
        for i, action_from in enumerate(self.actions):
            external_recommendation = self.external_minimizer[
                action_from
            ].recommendation(iteration, force)
            recommendations[i, :] = [
                external_recommendation[action_to]
                for j, action_to in enumerate(self.actions)
            ]
        # compute the stationary distribution of the markov chain transition matrix determined by the rec. matrix.
        # This will be the eigenvector to the eigenvalue 1.
        eigenvalues, eigenvectors = np.linalg.eig(recommendations.transpose())
        rec = eigenvectors[np.where(np.isclose(eigenvalues, 1.0))[0]]
        self.recommendation = rec / rec.sum()
        self._recommendation_computed = True
