from abc import ABC, abstractmethod
from typing import (
    Dict,
    Callable,
    Optional,
    Type,
)

import numpy as np

from cfrainbow.spiel_types import *
from cfrainbow.utils import slice_kwargs
from .external import ExternalRegretMinimizer


class InternalRegretMinimizer(ABC):
    def __init__(self, actions: Iterable[Action], *args, **kwargs):
        self.recommendation: Dict[Action, Probability] = {}
        self._actions = list(actions)
        self._n_actions = len(self.actions)
        self._recommendation_computed: bool = False
        self._last_update_time: int = -1

    def __len__(self):
        return self._n_actions

    @property
    def actions(self):
        return self._actions

    @property
    def last_update_time(self):
        return self._last_update_time

    def reset(self):
        self._reset()
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

    @abstractmethod
    def _reset(self, *args, **kwargs):
        pass

    @abstractmethod
    def _recommend(
        self, iteration: Optional[int] = None, *args, **kwargs
    ) -> Dict[Action, Probability]:
        raise NotImplementedError(
            f"method '{self._recommend.__name__}' is not implemented."
        )

    @abstractmethod
    def observe(
        self,
        iteration: int,
        utility: Callable[[Action, Action], float],
        *args,
        **kwargs,
    ):
        raise NotImplementedError(
            f"method '{self.observe.__name__}' is not implemented."
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

    def __len__(self):
        return self._n_actions

    def _reset(self):
        for erm in self.external_minimizer.values():
            erm.reset()

    @property
    def actions(self):
        return self._actions

    @property
    def last_update_time(self):
        return self._last_update_time

    @property
    def regret_mode(self):
        return self.external_minimizer[self.actions[0]].regret_mode

    def reset(self):
        for minimizer in self.external_minimizer.values():
            minimizer.reset()
        self.recommendation.clear()
        self._recommendation_computed = False
        self._last_update_time: int = -1

    def observe(
        self, iteration: int, utility: Callable[[Action], float], *args, **kwargs
    ):
        for assigned_action, erm in self.external_minimizer.items():
            erm.observe_utility(
                iteration,
                lambda a: utility(a) * self.recommendation[a],
            )
        self._last_update_time = iteration
        self._recommendation_computed = False

    def _recommend(self, iteration: int = None, force: bool = False, *args, **kwargs):
        n_actions = len(self)
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
        # compute the stationary distribution of the markov chain transition matrix
        # determined by the recommendation matrix.
        # This will be the eigenvector to the eigenvalue 1.
        eigenvalues, eigenvectors = np.linalg.eig(recommendations.transpose())
        rec = eigenvectors[np.where(np.isclose(eigenvalues, 1.0))[0]]
        self.recommendation = rec / rec.sum()
        self._recommendation_computed = True
