import inspect
from typing import Optional, Type, Union

import pyspiel
from open_spiel.python.algorithms import exploitability
from tqdm import tqdm

import cfrainbow.cfr as cfr
import cfrainbow.rm as rm
from cfrainbow.cfr.cfr_base import CFRBase

from cfrainbow.utils import (
    to_pyspiel_policy,
    normalize_policy_profile,
    slice_kwargs,
    load_game,
    PolicyPrinter,
    KuhnPolicyPrinter,
    make_uniform_policy,
)


def run(
    solver: Type[CFRBase],
    n_iter: int,
    regret_minimizer: Type[rm.ExternalRegretMinimizer] = rm.RegretMatcher,
    game: Union[pyspiel.Game, str] = "kuhn_poker",
    policy_printer: Optional[PolicyPrinter] = None,
    progressbar: bool = False,
    final_expl_print: bool = False,
    expl_threshold: Optional[float] = None,
    expl_check_freq: int = 1,
    **kwargs,
):
    # get all kwargs that can be found in the parent classes' and the given class's __init__ func
    solver_kwargs = slice_kwargs(
        kwargs, *[cls.__init__ for cls in inspect.getmro(solver)]
    )
    do_print = policy_printer is not None
    if do_print:
        print(
            f"Running {solver.__name__} "
            f"with regret minimizer {regret_minimizer.__name__} "
            f"and kwargs {solver_kwargs} "
            f"for {n_iter} iterations."
        )

    expl_values = []
    game = load_game(game)
    root_state = game.new_initial_state()
    uniform_joint_policy = make_uniform_policy(root_state)
    n_infostates = len(uniform_joint_policy)

    solver_obj = solver(
        root_state,
        regret_minimizer,
        verbose=do_print and not progressbar,
        **solver_kwargs,
    )

    gen = range(n_iter)
    for iteration in gen if not progressbar else (pbar := tqdm(gen)):
        if progressbar:
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

        solver_obj.iterate()

        avg_policy = solver_obj.average_policy()
        if iteration % expl_check_freq == 0:
            expl_values.append(
                exploitability.exploitability(
                    game,
                    to_pyspiel_policy(avg_policy, uniform_joint_policy),
                )
            )

            if do_print or (iteration == n_iter - 1 and final_expl_print):
                printed_profile = policy_printer.print_profile(
                    normalize_policy_profile(avg_policy)
                )
                if printed_profile:
                    print(
                        f"-------------------------------------------------------------"
                        f"--> Exploitability "
                        f"{f'{expl_values[-1]: .5f}' if 1e-5 < expl_values[-1] < 1e7 else f'{expl_values[-1]: .3E}'}"
                    )
                    print(printed_profile)
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
    avg_policy = solver_obj.average_policy()
    if (do_print or final_expl_print) and sum(
        map(lambda p: len(p), avg_policy)
    ) == n_infostates:
        print(policy_printer.print_profile(solver_obj.average_policy()))

    return expl_values


if __name__ == "__main__":
    n_iters = 10000
    for minimizer in (rm.RegretMatcher,):
        run(
            cfr.VanillaCFR,
            n_iters,
            game="kuhn_poker",
            regret_minimizer=minimizer,
            alternating=True,
            policy_printer=KuhnPolicyPrinter(),
            # progressbar=True,
            final_expl_print=False,
            expl_threshold=1e-8,
            seed=0,
            expl_check_freq=1,
        )
