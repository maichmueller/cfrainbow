from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Type

import numpy as np

from cfrainbow.spiel_types import *
from cfrainbow.utils import slice_kwargs

from .external import ExternalRegretMinimizer


class InternalRegretMinimizer(ABC):
    def __init__(self, actions: Union[int, Iterable[Action]], *args, **kwargs):
        self.recommendation: Dict[Action, Probability] = {}
        self._actions = (
            list(actions) if isinstance(actions, Iterable) else list(range(actions))
        )
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
        utility: Callable[[Action], float],
        *args,
        **kwargs,
    ):
        raise NotImplementedError(
            f"method '{self.observe.__name__}' is not implemented."
        )


class InternalFromExternalRegretMinimizer(InternalRegretMinimizer):
    def __init__(
        self,
        actions: Union[int, Iterable[Action]],
        external_regret_minimizer_factory: Callable[
            [List[Action]], ExternalRegretMinimizer
        ],
        *args,
        **kwargs,
    ):
        super().__init__(actions, *args, **kwargs)
        self.external_minimizer = {
            action: external_regret_minimizer_factory(self.actions)
            for action in self.actions
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

    def observe(
        self, iteration: int, utility: Callable[[Action], float], *args, **kwargs
    ):
        utility = {action: utility(action) for action in self.actions}
        for assigned_action, erm in self.external_minimizer.items():
            erm.observe(
                iteration,
                lambda action: utility[action] * self.recommendation[assigned_action],
            )
        self._last_update_time = iteration
        self._recommendation_computed = False

    def _recommend(self, iteration: int = None, force: bool = False, *args, **kwargs):
        """
        This method builds the recommendation matrix Q first and then proceeds to generate a (left) fix-point p = pQ.
        This fix-point is equivalently found as the eigenvector to the eigenvalue 1 when interpreting the
        recommendation matrix Q^T as a Markov Process and considering Q^T p^T = p^T (^T means transpose)
        """
        n_actions = len(self)
        # build the recommendations matrix
        recommendations = np.empty(shape=(n_actions, n_actions), dtype=float)
        for i, external_minimizer in enumerate(self.external_minimizer.values()):
            external_rec = external_minimizer.recommend(iteration, force)
            # external-regret minimizer returns a dictionary indexed by the actions
            recommendations[i, :] = [external_rec[action] for action in self.actions]
        # compute the stationary distribution of the markov chain transition matrix
        # determined by the recommendation matrix.
        # This will be the eigenvector to the eigenvalue 1.
        eigenvalues, eigenvectors = np.linalg.eig(recommendations.transpose())
        eigenvalue_1_index = np.where(np.isclose(eigenvalues, 1.0))[0]
        # eigenvectors[:, k] is the k-th eigenvector, not eigenvectors[k, :] !
        rec = np.real(eigenvectors[:, eigenvalue_1_index]).flatten()
        self.recommendation = rec / rec.sum()
        self._recommendation_computed = True
