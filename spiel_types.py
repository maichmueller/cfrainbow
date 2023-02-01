from __future__ import annotations
import itertools
from typing import Tuple, Mapping, Set, Union, Sequence
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
NormalFormStrategySpace = Set[NormalFormPlan]

# a sequence-form deterministic strategy (plan) is equivalent to a reduced normal-form plan
SequenceFormPlan = NormalFormPlan


class JointNormalFormPlan:
    def __init__(self, plans: Sequence[NormalFormPlan]):
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


class NumbaTypes:
    Action = nb.types.int32
    Probability = nb.types.float64
    Regret = nb.types.float64
    Value = nb.types.float64
    Infostate = nb.types.string
