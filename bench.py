import operator
from collections import namedtuple
from functools import singledispatch, reduce
from timeit import default_timer as timer
from typing import List, Union

import numpy as np
import pyspiel
import numba


import cmath
import itertools
from copy import copy
from typing import Sequence, Dict, Mapping, Tuple, Set, Optional

import pyspiel

from rm import InternalFromExternalRegretMinimizer, RegretMatcher
from type_aliases import Infostate, Action
from utils import all_states_gen, infostates_gen
from equilibria import (
    normal_form_strategy_space,
    normal_form_expected_payoff_table,
    normal_form_expected_payoff,
    cce_deviation_incentive,
    ce_deviation_incentive,
)


def timeit(func, *args, repeats: int = 1000, **kwargs):
    times = []
    for _ in range(repeats):
        s = timer()
        func(*args, **kwargs)
        e = timer()
        times.append(e - s)
    return sum(times) / len(times)


def main():
    # game = pyspiel.load_game("kuhn_poker(players=3)")
    game = pyspiel.load_game_as_turn_based("matching_pennies_3p")
    root = game.new_initial_state()
    spaces = normal_form_strategy_space(game)
    payoff_table = normal_form_expected_payoff_table(game, tuple(spaces.values()))
    strategy_space_size = {p: len(space) for p, space in spaces.items()}
    distr = {
        p: np.arange(1, len(space) + 1) / np.arange(1, len(space) + 1).sum()
        for p, space in spaces.items()
    }
    distr = {
        tuple(p[1] for p in plan): reduce(
            operator.mul,
            [
                distr[player][i]
                for (player, i) in zip(spaces.keys(), [p[0] for p in plan])
            ],
            1.0,
        )
        for plan in itertools.product(*[enumerate(space) for space in spaces.values()])
    }
    cce_epsilon = cce_deviation_incentive(distr, tuple(spaces.values()), payoff_table)
    ce_epsilon = ce_deviation_incentive(distr, tuple(spaces.values()), payoff_table)
    for p, space in spaces.items():
        print(p, ":", space)
    print(cce_epsilon)
    print(ce_epsilon)
    cce_epsilon = cce_deviation_incentive(distr, tuple(spaces.values()), payoff_table)
    ce_epsilon = ce_deviation_incentive(distr, tuple(spaces.values()), payoff_table)
    minim = []
    for player, space in spaces.items():
        minim.append(InternalFromExternalRegretMinimizer(space, RegretMatcher))

    # print("Jitted    :", timeit(normal_form_strategy_space, game))
    # print("Non-Jitted:", timeit(normal_form_strategy_space_nonjit, game))


if __name__ == "__main__":
    main()
