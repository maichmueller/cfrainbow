import itertools
import operator
from functools import reduce

import pyspiel
import pytest
from tqdm import tqdm

from cfrainbow import rm
from cfrainbow.cfr import *
from cfrainbow.cfr.cfr_base import CFRBase
from cfrainbow.utils import all_states_gen, to_pyspiel_tab_policy

MAX_ITER = int(1e6)
expl_threshold = 1e-2


@pytest.fixture(params=["python_efce_example_efg", "kuhn_poker"])
def setup_game(request):
    game = pyspiel.load_game(request.param)
    root_state = game.new_initial_state()

    all_infostates = set()
    avg_policy_list = {p: dict() for p in range(game.num_players())}
    uniform_joint_policy = dict()
    for state, _ in all_states_gen(root=root_state.clone()):
        infostate = state.information_state_string(state.current_player())
        actions = state.legal_actions()
        all_infostates.add(infostate)
        avg_policy_list[state.current_player()][infostate] = {
            action: 1.0 / len(actions) for action in actions
        }
        uniform_joint_policy[infostate] = [
            (action, 1.0 / len(actions)) for action in actions
        ]

    return game, root_state, all_infostates, list(avg_policy_list.values()), uniform_joint_policy


def case_name(val):
    if isinstance(val, CFRBase):
        return val.__name__
    elif isinstance(val, tuple):
        return f"regret_minimizer={val[0].__name__}"
    elif isinstance(val, dict):
        return ", ".join([f"{key}={value}" for key, value in val.items()])


@pytest.mark.parametrize(
    "cfr_class, cfr_class_args, cfr_class_kwargs",
    [
        (CFRVanilla, (rm.RegretMatcher,), dict(alternating=True)),
        (CFRVanilla, (rm.RegretMatcher,), dict(alternating=False)),
        (CFRPlus, (rm.RegretMatcherPlus,), dict()),
        (DiscountedCFR, (rm.RegretMatcherDiscounted,), dict(alternating=True)),
        (DiscountedCFR, (rm.RegretMatcherDiscounted,), dict(alternating=False)),
        (DiscountedCFR, (rm.RegretMatcherDiscountedPlus,), dict(alternating=True)),
        (DiscountedCFR, (rm.RegretMatcherDiscountedPlus,), dict(alternating=False)),
        (LinearCFR, (rm.RegretMatcherDiscounted,), dict(alternating=True)),
        (LinearCFR, (rm.RegretMatcherDiscounted,), dict(alternating=False)),
        (LinearCFR, (rm.RegretMatcherDiscountedPlus,), dict(alternating=True)),
        (LinearCFR, (rm.RegretMatcherDiscountedPlus,), dict(alternating=False)),
        (ExponentialCFR, (rm.RegretMatcher,), dict(alternating=True)),
        (ExponentialCFR, (rm.RegretMatcher,), dict(alternating=False)),
        (ExponentialCFR, (rm.RegretMatcherPlus,), dict(alternating=True)),
        (ExponentialCFR, (rm.RegretMatcherPlus,), dict(alternating=False)),
        (PredictiveCFRPlus, (rm.AutoPredictiveRegretMatcherPlus,), dict()),
    ]
    + [
        (
            OutcomeSamplingMCCFR,
            (minimizer,),
            dict(alternating=alternating, weighting_mode=mode, seed=0),
        )
        for minimizer, alternating, mode in itertools.product(
            (rm.RegretMatcher, rm.RegretMatcherPlus),
            (True, False),
            OutcomeSamplingWeightingMode,
        )
    ]
    + [
        (
            ExternalSamplingMCCFR,
            (minimizer,),
            dict(seed=0),
        )
        for minimizer in (rm.RegretMatcher, rm.RegretMatcherPlus)
    ]
    + [
        (
            ChanceSamplingCFR,
            (minimizer,),
            dict(alternating=alternating, seed=0),
        )
        for minimizer, alternating in itertools.product(
            (rm.RegretMatcher, rm.RegretMatcherPlus), (True, False)
        )
    ]
    + [
        (
            PureCFR,
            (minimizer,),
            dict(alternating=alternating, seed=0),
        )
        for minimizer, alternating in itertools.product(
            (rm.RegretMatcher, rm.RegretMatcherPlus), (True, False)
        )
    ],
    ids=case_name,
)
def test_efg(
    setup_game,
    cfr_class,
    cfr_class_args,
    cfr_class_kwargs,
):
    game, root_state, all_infostates, avg_policy_list, uniform_joint = setup_game
    solver = cfr_class(
        root_state,
        *cfr_class_args,
        verbose=False,
        # average_policy_list=avg_policy_list,
        **cfr_class_kwargs,
    )

    for i in tqdm(range(MAX_ITER)):
        solver.iterate()

        avg_policy = solver.average_policy()

        if (i % 1000) == 0:
            expl_value = pyspiel.exploitability(
                game,
                to_pyspiel_tab_policy(avg_policy, uniform_joint),
            )
            if expl_value < expl_threshold:
                return
    assert False