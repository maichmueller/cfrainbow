import itertools
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import pyspiel
from matplotlib import pyplot as plt
from cfr import CFR
from cfr_linear import LinearCFR
from cfr_plus import CFRPlus
from cfr_discounted import DiscountedCFR
from cfr_exp import ExponentialCFR

from multiprocessing import Pool, freeze_support, cpu_count
from open_spiel.python.algorithms import exploitability

from utils import (
    to_pyspiel_tab_policy,
    print_final_policy_profile,
    print_policy_profile,
)


def plot_cfr_convergence(
    algorithm_to_expl_lists: Dict[str, List[float]],
    game_name: str = "Kuhn Poker",
    save: bool = False,
    save_name: Optional[str] = None,
):
    with plt.style.context("bmh"):
        max_iters = 0
        plt.figure()

        for algo, expl_list in algorithm_to_expl_lists.items():
            max_iters = max(max_iters, len(expl_list))
            linestyle = "--" if "(A)" in algo else "-"
            plt.plot(expl_list, label=algo, linewidth=0.75, linestyle=linestyle)

        plt.xlabel("Iteration")
        plt.ylabel("Exploitability")
        plt.yscale("log", base=10)
        plt.title(f"Convergence to Nash Equilibrium in {game_name}")
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            fancybox=True,
            shadow=True,
            ncol=4,
        )

        if not save:
            plt.show(bbox_inches="tight")
        else:
            plt.savefig(f"{save_name}_iters_{max_iters}.pdf", bbox_inches="tight")


def running_mean(values, window_size: int = 10):
    # return np.convolve(values, np.ones(window_size) / window_size, mode="valid")
    return [
        np.mean(values[max(0, -window_size + i) : i + 1]) for i in range(len(values))
    ]


def main(
    cfr_class,
    n_iter: int,
    game_name: str = "kuhn_poker",
    do_print: bool = True,
    **kwargs,
):
    if do_print:
        print(
            f"Running class {cfr_class.__name__} with kwargs {kwargs} for {n_iter} iterations."
        )

    expl_values = []
    game = pyspiel.load_game(game_name)
    root_state = game.new_initial_state()
    n_players = list(range(root_state.num_players()))
    current_policies = [{} for _ in n_players]
    average_policies = [{} for _ in n_players]
    solver = cfr_class(
        root_state, current_policies, average_policies, **kwargs, verbose=do_print,
    )
    simultaneous_updates = kwargs.get("simultaneous_updates", False)
    for i in range(n_iter):
        solver.iterate()

        if simultaneous_updates or (not simultaneous_updates and i > 1):
            avg_policy = solver.average_policy()
            expl_values.append(
                exploitability.exploitability(
                    game, to_pyspiel_tab_policy(avg_policy),
                )
            )

            if do_print:
                print(
                    f"-------------------------------------------------------------"
                    f"--> Exploitability {expl_values[-1]: .5f}"
                )
                if game_name == "kuhn_poker":
                    print_policy_profile(deepcopy(avg_policy))
                    print(
                        f"---------------------------------------------------------------"
                    )
    if do_print and game_name == "kuhn_poker":
        print_final_policy_profile(solver.average_policy())

    return expl_values


def main_wrapper(args):
    name, pos_args, kwargs = args
    return name, main(*pos_args, **kwargs)


if __name__ == "__main__":
    n_iters = 1000
    verbose = False
    game = "kuhn_poker"
    # game = "leduc_poker"
    with Pool(processes=cpu_count()) as pool:
        jobs = {
            "CFR (A)": (
                CFR,
                n_iters,
                {"game_name": game, "simultaneous_updates": False, "do_print": verbose},
            ),
            "CFR (S)": (
                CFR,
                n_iters,
                {"game_name": game, "simultaneous_updates": True, "do_print": verbose},
            ),
            "Exp. CFR (A)": (
                CFR,
                n_iters,
                {"game_name": game, "simultaneous_updates": False, "do_print": verbose},
            ),
            "Exp. CFR (S)": (
                CFR,
                n_iters,
                {"game_name": game, "simultaneous_updates": True, "do_print": verbose},
            ),
            "CFR+.": (CFRPlus, n_iters, {"game_name": game, "do_print": verbose}),
            "Disc. CFR (A)": (
                DiscountedCFR,
                n_iters,
                {"game_name": game, "simultaneous_updates": False, "do_print": verbose},
            ),
            "Disc. CFR (S)": (
                DiscountedCFR,
                n_iters,
                {"game_name": game, "simultaneous_updates": True, "do_print": verbose},
            ),
            "Disc. CFR+ (A)": (
                DiscountedCFR,
                n_iters,
                {
                    "game_name": game,
                    "simultaneous_updates": False,
                    "do_regret_matching_plus": True,
                    "do_print": verbose,
                },
            ),
            "Disc. CFR+ (S)": (
                DiscountedCFR,
                n_iters,
                {
                    "game_name": game,
                    "simultaneous_updates": True,
                    "do_regret_matching_plus": True,
                    "do_print": verbose,
                },
            ),
            "Lin. CFR (A)": (
                LinearCFR,
                n_iters,
                {"game_name": game, "simultaneous_updates": False, "do_print": verbose},
            ),
            "Lin. CFR (S)": (
                LinearCFR,
                n_iters,
                {"game_name": game, "simultaneous_updates": True, "do_print": verbose},
            ),
            "Lin. CFR+ (A)": (
                LinearCFR,
                n_iters,
                {
                    "game_name": game,
                    "simultaneous_updates": False,
                    "do_regret_matching_plus": True,
                    "do_print": verbose,
                },
            ),
            "Lin. CFR+ (S)": (
                LinearCFR,
                n_iters,
                {
                    "game_name": game,
                    "simultaneous_updates": True,
                    "do_regret_matching_plus": True,
                    "do_print": verbose,
                },
            ),
        }
        results = pool.map_async(
            main_wrapper,
            [
                (name, args_and_kwargs[:-1], args_and_kwargs[-1])
                for name, args_and_kwargs in jobs.items()
            ],
        )
        expl_dict = {}
        for result in results.get():
            name, values = result
            expl_dict[name] = values

    averaged_values = {
        name: running_mean(expl_values, window_size=20)
        for name, expl_values in expl_dict.items()
    }
    plot_cfr_convergence(
        averaged_values,
        game_name=" ".join([s.capitalize() for s in game.split("_")]),
        save=True,
        save_name=f"{game}",
    )
