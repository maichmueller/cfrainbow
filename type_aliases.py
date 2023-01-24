from typing import Tuple, Mapping, Set, Union


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
