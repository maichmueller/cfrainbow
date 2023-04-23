import inspect
import operator
from functools import reduce
from typing import Optional, Type, Union

import pyspiel
from open_spiel.python.algorithms import exploitability
from tqdm import tqdm

import cfrainbow.cfr as cfr
import cfrainbow.rm as rm
from cfrainbow.cfr.cfr_base import CFRBase

from cfrainbow.utils import (
    all_states_gen,
    print_final_policy_profile,
    print_kuhn_poker_policy_profile,
    to_pyspiel_policy,
    normalize_policy_profile,
    slice_kwargs,
)


def run(
    solver: Type[CFRBase],
    n_iter: int,
    regret_minimizer: type[rm.ExternalRegretMinimizer] = rm.RegretMatcher,
    game: Union[pyspiel.Game, str] = "kuhn_poker",
    do_print: bool = True,
    tqdm_print: bool = False,
    only_final_expl_print: bool = False,
    expl_threshold: Optional[float] = None,
    expl_check_freq: int = 1,
    **kwargs,
):
    # get all kwargs that can be found in the parent classes' and the given class's __init__ func
    solver_kwargs = slice_kwargs(
        kwargs, *[cls.__init__ for cls in inspect.getmro(solver)]
    )
    if do_print:
        print(
            f"Running {solver.__name__} "
            f"with regret minimizer {regret_minimizer.__name__} "
            f"and kwargs {solver_kwargs} "
            f"for {n_iter} iterations."
        )

    expl_values = []
    game = pyspiel.load_game(game)
    root_state = game.new_initial_state()
    all_infostates = set()
    uniform_joint_policy = dict()
    for state, _ in all_states_gen(root=root_state.clone()):
        infostate = state.information_state_string(state.current_player())
        actions = state.legal_actions()
        all_infostates.add(infostate)
        uniform_joint_policy[infostate] = [
            (action, 1.0 / len(actions)) for action in actions
        ]

    n_infostates = len(all_infostates)

    solver = solver(
        root_state,
        regret_minimizer,
        verbose=do_print and not tqdm_print,
        **solver_kwargs,
    )

    gen = range(n_iter)
    for iteration in gen if not tqdm_print else (pbar := tqdm(gen)):
        if tqdm_print:
            if expl_values:
                expl_print = f"{f'{expl_values[-1]: .5f}' if expl_values and expl_values[-1] > 1e-5 else f'{expl_values[-1]: .3E}'}"
            else:
                expl_print = " - "
            pbar.set_description(
                f"Method:{solver.__name__} | "
                f"RM:{regret_minimizer.__name__} | "
                f"kwargs:{solver_kwargs} | "
                f"{expl_print}"
            )

        solver.iterate()

        avg_policy = solver.average_policy()
        if iteration % expl_check_freq == 0:
            expl_values.append(
                exploitability.exploitability(
                    game,
                    to_pyspiel_policy(avg_policy, uniform_joint_policy),
                )
            )

            if (do_print and not only_final_expl_print) or (
                iteration == n_iter - 1 and only_final_expl_print
            ):
                print(
                    f"-------------------------------------------------------------"
                    f"--> Exploitability "
                    f"{f'{expl_values[-1]: .5f}' if 1e-5 < expl_values[-1] < 1e7  else f'{expl_values[-1]: .3E}'}"
                )
                if str(game) == "kuhn_poker()":
                    print_kuhn_poker_policy_profile(
                        normalize_policy_profile(avg_policy)
                    )
                    print(
                        f"---------------------------------------------------------------"
                    )

            if expl_threshold is not None and expl_values[-1] < expl_threshold:
                print(f"Exploitability threshold of {expl_threshold} reached.")
                break
    if do_print:
        print(
            "\n---------------------------------------------------------------> Final Exploitability:",
            expl_values[-1],
        )
    avg_policy = solver.average_policy()
    if (
        (do_print or only_final_expl_print)
        and str(game) == "kuhn_poker()"
        and sum(map(lambda p: len(p), avg_policy)) == n_infostates
    ):
        print_final_policy_profile(solver.average_policy())

    return expl_values


if __name__ == "__main__":
    # n_iters = 10000
    n_iters = int(1e10)
    for minimizer in (rm.RegretMatcher,):
        run(
            cfr.CFRVanilla,
            n_iters,
            regret_minimizer=minimizer,
            alternating=True,
            do_print=False,
            tqdm_print=True,
            only_final_expl_print=False,
            # weighting_mode=cfr.OutcomeSamplingWeightingMode.lazy,
            expl_threshold=1e-8,
            seed=0,
            # game_name="python_efce_example_efg",
            game="kuhn_poker",
            expl_check_freq=1,
        )
