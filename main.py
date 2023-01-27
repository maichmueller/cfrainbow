import pyspiel
from open_spiel.python.algorithms import exploitability
from tqdm import tqdm

import rm
import cfr

from utils import (
    all_states_gen,
    print_final_policy_profile,
    print_kuhn_poker_policy_profile,
    to_pyspiel_tab_policy,
    normalize_policy_profile,
    slice_kwargs,
)
import inspect


def main(
    n_iter: int,
    cfr_class,
    game_name: str = "kuhn_poker",
    do_print: bool = True,
    **kwargs,
):
    solver_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in inspect.signature(cfr_class.__init__).parameters
    }

    if do_print:
        print(
            f"Running class {cfr_class.__name__} with kwargs {solver_kwargs} for {n_iter} iterations."
        )

    game = pyspiel.load_game(game_name)
    root_state = game.new_initial_state()
    all_infostates = {
        state.information_state_string(state.current_player())
        for state, _ in all_states_gen(root=root_state.clone())
    }
    n_infostates = len(all_infostates)
    n_players = list(range(root_state.num_players()))

    solver = cfr_class(
        root_state,
        curr_policy_list=[{} for _ in n_players],
        average_policy_list=[{} for _ in n_players],
        verbose=do_print,
        **solver_kwargs,
    )

    expl_values = []
    for i in range(n_iter):
        solver.iterate()

        if sum(map(lambda p: len(p), solver.average_policy())) == n_infostates:
            avg_policy = solver.average_policy()
            expl_values.append(
                exploitability.exploitability(
                    game,
                    to_pyspiel_tab_policy(avg_policy),
                )
            )

            if do_print:
                print(
                    f"-------------------------------------------------------------"
                    f"--> Exploitability {expl_values[-1]: .5f}"
                )
                if game_name == "kuhn_poker":
                    print_kuhn_poker_policy_profile(
                        normalize_policy_profile(avg_policy)
                    )
                    print(
                        f"---------------------------------------------------------------"
                    )
    avg_policy = solver.average_policy()
    if (
        do_print
        and game_name == "kuhn_poker"
        and sum(map(lambda p: len(p), solver.average_policy())) == n_infostates
    ):
        print_final_policy_profile(avg_policy)

    return expl_values


def main_nash(
    cfr_class,
    n_iter,
    regret_minimizer: type[rm.ExternalRegretMinimizer] = rm.RegretMatcher,
    game_name: str = "kuhn_poker",
    do_print: bool = True,
    tqdm_print: bool = False,
    only_final_expl_print: bool = False,
    **kwargs,
):
    # get all kwargs that can be found in the parent classes' and the given class's __init__ func
    solver_kwargs = slice_kwargs(
        kwargs, *[cls.__init__ for cls in inspect.getmro(cfr_class)]
    )
    if do_print:
        print(
            f"Running {cfr_class.__name__} "
            f"with regret minimizer {regret_minimizer.__name__} "
            f"and kwargs {solver_kwargs} "
            f"for {n_iter} iterations."
        )

    expl_values = []
    game = pyspiel.load_game(game_name)
    root_state = game.new_initial_state()
    all_infostates = {
        state.information_state_string(state.current_player())
        for state, _ in all_states_gen(root=root_state.clone())
    }
    n_infostates = len(all_infostates)

    solver = cfr_class(
        root_state,
        regret_minimizer,
        verbose=do_print and not tqdm_print,
        **solver_kwargs,
    )

    gen = range(n_iter)
    for i in gen if not tqdm_print else tqdm(gen):
        solver.iterate()

        avg_policy = solver.average_policy()
        if sum(map(lambda p: len(p), avg_policy)) == n_infostates:
            expl_values.append(
                exploitability.exploitability(
                    game,
                    to_pyspiel_tab_policy(avg_policy),
                )
            )

            if ((do_print or tqdm_print) and not only_final_expl_print) or (
                i == n_iter - 1 and only_final_expl_print
            ):
                print(
                    f"-------------------------------------------------------------"
                    f"--> Exploitability {expl_values[-1]: .5f}"
                )
                if game_name == "kuhn_poker":
                    print_kuhn_poker_policy_profile(
                        normalize_policy_profile(avg_policy)
                    )
                    print(
                        f"---------------------------------------------------------------"
                    )

    avg_policy = solver.average_policy()
    if (
        (do_print or only_final_expl_print)
        and game_name == "kuhn_poker"
        and sum(map(lambda p: len(p), avg_policy)) == n_infostates
    ):
        print_final_policy_profile(solver.average_policy())

    return expl_values


def main_cce(
    cfr_class,
    n_iter,
    regret_minimizer: type[rm.ExternalRegretMinimizer] = rm.RegretMatcher,
    game_name: str = "kuhn_poker",
    do_print: bool = True,
    tqdm_print: bool = False,
    only_final_expl_print: bool = False,
    **kwargs,
):
    # get all kwargs that can be found in the parent classes' and the given class's __init__ func
    # possible_kwargs = {}
    # for cls in inspect.getmro(cfr_class):
    #     possible_kwargs |= inspect.signature(cls.__init__).parameters
    # solver_kwargs = {k: v for k, v in kwargs.items() if k in possible_kwargs}
    solver_kwargs = slice_kwargs(
        kwargs, *[cls.__init__ for cls in inspect.getmro(cfr_class)]
    )

    if do_print:
        print(
            f"Running {cfr_class.__name__} "
            f"with regret minimizer {regret_minimizer.__name__} "
            f"and kwargs {solver_kwargs} "
            f"for {n_iter} iterations."
        )

    expl_values = []
    game = pyspiel.load_game(game_name)
    root_state = game.new_initial_state()
    all_infostates = {
        state.information_state_string(state.current_player())
        for state in all_states_gen(root=root_state.clone())
    }
    n_infostates = len(all_infostates)

    solver = cfr_class(
        root_state,
        regret_minimizer,
        verbose=do_print and not tqdm_print,
        **solver_kwargs,
    )

    gen = range(n_iter)
    for i in gen if not tqdm_print else tqdm(gen):
        solver.iterate()

        avg_policy = solver.average_policy()
        if sum(map(lambda p: len(p), avg_policy)) == n_infostates:
            expl_values.append(
                exploitability.exploitability(
                    game,
                    to_pyspiel_tab_policy(avg_policy),
                )
            )

            if ((do_print or tqdm_print) and not only_final_expl_print) or (
                i == n_iter - 1 and only_final_expl_print
            ):
                print(
                    f"-------------------------------------------------------------"
                    f"--> Exploitability {expl_values[-1]: .5f}"
                )
                if "kuhn_poker" in game_name:
                    print_kuhn_poker_policy_profile(
                        normalize_policy_profile(avg_policy)
                    )
                    print(
                        f"---------------------------------------------------------------"
                    )

    avg_policy = solver.average_policy()
    if (
        (do_print or only_final_expl_print)
        and "kuhn_poker" in game_name
        and sum(map(lambda p: len(p), avg_policy)) == n_infostates
    ):
        print_final_policy_profile(solver.average_policy())

    return expl_values


if __name__ == "__main__":
    n_iters = 10000
    for minimizer in (rm.RegretMatcherPredictivePlus,):
        main_nash(
            cfr.PredictivePlusCFR,
            n_iters,
            regret_minimizer=minimizer,
            alternating=True,
            do_print=True,
            tqdm_print=False,
            only_final_expl_print=False,
            weighting_mode=cfr.OutcomeSamplingWeightingMode.optimistic,
            seed=0,
        )
