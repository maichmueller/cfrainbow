from __future__ import annotations
import itertools
from typing import Tuple, Mapping, Set, Union, Sequence, Iterable
import numba as nb

Player = int
Action = int
Probability = float
Regret = float
Value = float
Infostate = str

# a plan is a deterministic strategy
NormalFormPlan = Tuple[Tuple[Infostate, Action], ...]
# a strategy is a distribution over plans
NormalFormStrategy = Mapping[NormalFormPlan, float]
# a normal-form strategy space is a set of strategies
NormalFormStrategySpace = Set[NormalFormPlan]

# a sequence-form deterministic strategy (plan) is equivalent to a reduced normal-form plan
SequenceFormPlan = NormalFormPlan
# a sequence-form strategy is a distribution over sequence-form plans
SequenceFormStrategy = NormalFormStrategy
# a sequence-form strategy space is a set of sequence-form strategies
SequenceFormStrategySpace = Set[NormalFormStrategy]


class JointNormalFormPlan:
    def __init__(self, plans: Iterable[NormalFormPlan]):
        self.plans: tuple[NormalFormPlan] = tuple(plans)
        self.hash = hash(tuple(elem for elem in itertools.chain(self.plans)))

    def __contains__(self, item):
        return any(item in plan for plan in self.plans)

    def __hash__(self):
        return self.hash

    def __eq__(self, other: Union[Sequence[NormalFormPlan], JointNormalFormPlan]):
        if isinstance(other, JointNormalFormPlan):
            return self.plans == other.plans
        return all(other_plan == plan for other_plan, plan in zip(other, self.plans))

    def __repr__(self):
        return repr(self.plans)


class JointNormalFormStrategy:
    def __init__(self, plans: Iterable[NormalFormStrategy]):
        self.strategies: tuple[NormalFormStrategy] = tuple(plans)
        # python float hashing distinguishes up to 1e-16 difference in the float
        self.hash = hash(
            tuple(itertools.chain(map(lambda s: s.items(), self.strategies)))
        )

    def __contains__(self, other_strategy):
        return any(other_strategy == strategy for strategy in self.strategies)

    def __hash__(self):
        return self.hash

    def __eq__(
        self, other: Union[Sequence[NormalFormStrategy], JointNormalFormStrategy]
    ):
        if isinstance(other, JointNormalFormStrategy):
            return self.strategies == other.strategies
        return all(
            other_strategy == strategy
            for other_strategy, strategy in zip(other, self.strategies)
        )

    def __repr__(self):
        return repr(self.strategies)
