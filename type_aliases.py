from typing import Tuple, Mapping, Set, Union


Action = int
Probability = float
Regret = float
Value = float
Infostate = str


NormalFormPlan = Union[Tuple[Infostate, Action], Action]
NormalFormStrategy = Mapping[NormalFormPlan, float]
NormalFormStrategySpace = Set[NormalFormPlan]
