import operator
import os
from typing import Dict, List, Optional
import pickle
import matplotlib.cm
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from tqdm import tqdm

from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count

from cfrainbow import rm
from cfrainbow.main import main
from cfrainbow.cfr import *


def plot_cfr_convergence(
    iters_run: int,
    algorithm_to_expl_lists: Dict[str, List[List[float]]],
    game_name: str = "Kuhn Poker",
    save: bool = False,
    save_name: Optional[str] = None,
):
    with plt.style.context("bmh"):
        max_iters = 0
        plt.figure()
        cmap = matplotlib.cm.get_cmap("tab20")

        unpaired_algo_names = [
            [
                full_algo_name
                for full_algo_name in algorithm_to_expl_lists.keys()
                if algo in full_algo_name
            ][0]
            for algo, count in Counter(
                [
                    name.replace("(A)", "").replace("(S)", "")
                    for name in algorithm_to_expl_lists.keys()
                ]
            ).items()
            if count == 1
        ]
        unpaired_algo_list = sorted(
            [(algo, algorithm_to_expl_lists.pop(algo)) for algo in unpaired_algo_names],
            key=operator.itemgetter(0),
        )
        algorithm_to_expl_lists = (
            sorted(list(algorithm_to_expl_lists.items()), key=operator.itemgetter(0))
            + unpaired_algo_list
        )
        dash_pattern = matplotlib.rcParams["lines.dashed_pattern"]
        linewidth = 1
        x = np.arange(1, iters_run + 1)
        fig, ax = plt.subplots(figsize=(8, 8))
        manual_legend_handles = []
        for i, (algo, expl_list) in enumerate(algorithm_to_expl_lists):
            max_iters = max(max_iters, len(expl_list))
            linestyle = "--" if "(A)" in algo else "-"
            color = cmap(i)
            if len(expl_list) != 1:
                iteration_counts = np.flip(
                    np.sort(np.asarray([len(vals) for vals in expl_list])), 0
                )
                iter_beginnings = iters_run - iteration_counts
                freq_arr = np.asarray(
                    sorted(Counter(iter_beginnings).items(), key=operator.itemgetter(0)),
                    dtype=float,
                )
                absolute_freq = np.cumsum(freq_arr, axis=0)[:, 1]
                relative_freq = absolute_freq / len(expl_list)
                iter_buckets = freq_arr[:, 0].astype(int)

                iter_to_bucket = (
                    np.searchsorted(
                        iter_buckets,
                        x - 1,  # x is the 'logical' x scale to plot (iteration 1...T),
                        # but values are index starting from 0, so -1 to get the index scale
                        side="right",
                    )
                    - 1
                )
                no_value_mask = iter_to_bucket == -1
                no_value_iters = x[no_value_mask]

                absolute_freq_per_iter = np.concatenate(
                    [
                        np.zeros(no_value_iters.size),
                        absolute_freq[iter_to_bucket[~no_value_mask]],
                    ]
                )
                relative_freq_per_iter = np.concatenate(
                    [
                        np.zeros(no_value_iters.size),
                        relative_freq[iter_to_bucket[~no_value_mask]],
                    ]
                )

                padded_iter_values_matrix = (
                    torch.nn.utils.rnn.pad_sequence(
                        [  # full-length zeros tensor to force padding for the entire run_iters length
                            torch.zeros(iters_run)
                        ]
                        + [torch.Tensor(list(reversed(a))) for a in expl_list],
                        batch_first=True,
                    )
                    .flip(dims=(1,))
                    .numpy()
                )
                nanpadded_iter_values_matrix = np.where(
                    padded_iter_values_matrix == 0, np.nan, padded_iter_values_matrix
                )
                sum_band = padded_iter_values_matrix.sum(axis=0)
                mean_band = sum_band / absolute_freq_per_iter
                stderr = np.nanstd(
                    nanpadded_iter_values_matrix, ddof=1, axis=0
                ) / np.sqrt(np.isnan(nanpadded_iter_values_matrix).sum(axis=0))

                nan_mask = np.isnan(mean_band)
                x_to_plot = (x - 1)[~nan_mask] + 1
                # repackage the x, y data points as a list of 2D coordinates
                mean_to_plot = mean_band[~nan_mask]
                stderr_to_plot = stderr[~nan_mask]
                # interp_xspace = np.linspace(x_to_plot[0], iters_run, iters_run * 10)
                # interp_mean = np.interp(interp_xspace, x_to_plot, mean_to_plot)
                # interp_alpha = np.interp(interp_xspace, x_to_plot, relative_freq[iter_to_bucket[~no_value_mask]])
                # points = np.array([interp_xspace, interp_mean]).T.reshape(-1, 1, 2)
                points = np.array([x_to_plot, mean_to_plot]).T.reshape(-1, 1, 2)
                # we build a line collection which plots individual lines defined by a starting coordinate (x1,y1)
                # and a target coordinate (x2,y2). Thus, in order to make our function points y plotted, we interleave
                # our y points by having each coordinate point to its next coordinate as target. We then transform like:
                #    ([x1,y1],       (([x1,y1],[x2,y2]),
                #     [x2,y2],        ([x2,y2],[x3,y3]),
                #     [x3,y3],  -->   ([x3,y3],[x4,y4]),
                #     [x4,y4],        ([x4,y4],[x5,y5]))
                #     [x5,y5])
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                if linestyle == "--":
                    lc = LineCollection(
                        [s for i, s in enumerate(segments) if i % 19 > 7],
                        linewidths=relative_freq_per_iter * linewidth,
                        color=color,
                        alpha=relative_freq_per_iter,
                    )
                    # lc = LineCollection(
                    #     segments,
                    #     linewidths=relative_freq_per_iter * linewidth,
                    #     color=color,
                    #     alpha=relative_freq_per_iter,
                    # )
                    # ax.plot(
                    #     x_to_plot,
                    #     mean_to_plot,
                    #     linewidth=linewidth,
                    #     linestyle=linestyle,
                    #     color=color,
                    # )
                    ax.add_collection(lc)
                else:
                    lc = LineCollection(
                        segments,
                        linestyles=linestyle,
                        linewidths=relative_freq_per_iter * linewidth,
                        color=color,
                        alpha=relative_freq_per_iter**2,
                    )
                    ax.add_collection(lc)
                # this plot is only here to add the legend entry
                line = Line2D(
                    [0],
                    [0],
                    label=algo,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    color=color,
                )
                manual_legend_handles.append(line)
                #
                fill_area_polygon = ax.fill_between(
                    x_to_plot,
                    mean_to_plot - stderr_to_plot,
                    mean_to_plot + stderr_to_plot,
                    color=color,
                    alpha=0.1,
                )

        for i, (algo, expl_list) in enumerate(algorithm_to_expl_lists):
            color = cmap(i)
            if len(expl_list) == 1:
                ax.plot(
                    x[iters_run - len(expl_list[0]) :],
                    expl_list[0],
                    label=algo,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    color=color,
                )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Exploitability")
        ax.set_yscale("log", base=10)
        # ax.set_xscale("log", base=10)
        ax.set_title(f"Convergence to Nash Equilibrium in {game_name}")
        manual_legend_handles.extend(ax.get_legend_handles_labels()[0])
        ax.legend(
            handles=manual_legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            fancybox=True,
            shadow=True,
            ncol=4,
        )

        if not save:
            plt.show(bbox_inches="tight")
        else:
            plt.savefig(f"{save_name}_iters_{iters_run}.pdf", bbox_inches="tight")


def running_mean(values, window_size: int = 10):
    # return np.convolve(values, np.ones(window_size) / window_size, mode="valid")
    return [
        np.mean(values[max(0, -window_size + i) : i + 1]) for i in range(len(values))
    ]


def main_wrapper(args):
    name, pos_args, kwargs = args
    return name, main(*pos_args, **kwargs)


if __name__ == "__main__":
    n_iters = 1000
    verbose = False
    game = "kuhn_poker"
    # game = "leduc_poker"
    rng = np.random.default_rng(0)
    stochastic_seeds = 10
    n_cpu = cpu_count()
    filename = "testdata.pkl"
    if not os.path.exists(os.path.join(".", filename)):
        with Pool(processes=n_cpu) as pool:
            jobs = list(
                {
                    "CFR (A)": (
                        CFRVanilla,
                        n_iters,
                        {
                            "game_name": game,
                            "regret_minimizer": rm.RegretMatcher,
                            "alternating": True,
                            "do_print": verbose,
                        },
                    ),
                    "CFR (S)": (
                        CFRVanilla,
                        n_iters,
                        {
                            "game_name": game,
                            "regret_minimizer": rm.RegretMatcher,
                            "alternating": False,
                            "do_print": verbose,
                        },
                    ),
                    "Exp. CFR (A)": (
                        ExponentialCFR,
                        n_iters,
                        {
                            "game_name": game,
                            "regret_minimizer": rm.RegretMatcher,
                            "alternating": True,
                            "do_print": verbose,
                        },
                    ),
                    "Exp. CFR (S)": (
                        ExponentialCFR,
                        n_iters,
                        {
                            "game_name": game,
                            "regret_minimizer": rm.RegretMatcher,
                            "alternating": False,
                            "do_print": verbose,
                        },
                    ),
                    "Pure CFR (A)": (
                        PureCFR,
                        n_iters,
                        {
                            "game_name": game,
                            "regret_minimizer": rm.RegretMatcher,
                            "stochastic_solver": True,
                            "alternating": True,
                            "do_print": verbose,
                        },
                    ),
                    "Pure CFR (S)": (
                        PureCFR,
                        n_iters,
                        {
                            "game_name": game,
                            "regret_minimizer": rm.RegretMatcher,
                            "stochastic_solver": True,
                            "alternating": False,
                            "do_print": verbose,
                        },
                    ),
                    "OS-MCCFR (A)": (
                        OutcomeSamplingMCCFR,
                        n_iters,
                        {
                            "game_name": game,
                            "regret_minimizer": rm.RegretMatcher,
                            "stochastic_solver": True,
                            "weighting_mode": 2,
                            "alternating": True,
                            "do_print": verbose,
                        },
                    ),
                    "OS-MCCFR (S)": (
                        OutcomeSamplingMCCFR,
                        n_iters,
                        {
                            "game_name": game,
                            "regret_minimizer": rm.RegretMatcher,
                            "stochastic_solver": True,
                            "weighting_mode": 2,
                            "alternating": False,
                            "do_print": verbose,
                        },
                    ),
                    "ES-MCCFR (A)": (
                        ExternalSamplingMCCFR,
                        n_iters,
                        {
                            "game_name": game,
                            "regret_minimizer": rm.RegretMatcher,
                            "stochastic_solver": True,
                            "alternating": True,
                            "do_print": verbose,
                        },
                    ),
                    "CS-MCCFR (A)": (
                        ChanceSamplingCFR,
                        n_iters,
                        {
                            "game_name": game,
                            "regret_minimizer": rm.RegretMatcher,
                            "stochastic_solver": True,
                            "alternating": True,
                            "do_print": verbose,
                        },
                    ),
                    "CS-MCCFR (S)": (
                        ChanceSamplingCFR,
                        n_iters,
                        {
                            "game_name": game,
                            "regret_minimizer": rm.RegretMatcher,
                            "stochastic_solver": True,
                            "alternating": False,
                            "do_print": verbose,
                        },
                    ),
                    "CFR+ (A)": (
                        CFRPlus,
                        n_iters,
                        {"game_name": game,
                            "regret_minimizer": rm.RegretMatcherPlus, "do_print": verbose},
                    ),
                    "Disc. CFR (A)": (
                        DiscountedCFR,
                        n_iters,
                        {
                            "game_name": game,
                            "regret_minimizer": rm.RegretMatcherDiscounted,
                            "alternating": True,
                            "do_print": verbose,
                        },
                    ),
                    "Disc. CFR (S)": (
                        DiscountedCFR,
                        n_iters,
                        {
                            "game_name": game,
                            "regret_minimizer": rm.RegretMatcherDiscounted,
                            "alternating": False,
                            "do_print": verbose,
                        },
                    ),
                    "Disc. CFR+ (A)": (
                        DiscountedCFR,
                        n_iters,
                        {
                            "game_name": game,
                            "alternating": True,
                            "regret_minimizer": rm.RegretMatcherDiscountedPlus,
                            "do_print": verbose,
                        },
                    ),
                    "Disc. CFR+ (S)": (
                        DiscountedCFR,
                        n_iters,
                        {
                            "game_name": game,
                            "alternating": False,
                            "regret_minimizer": rm.RegretMatcherDiscountedPlus,
                            "do_print": verbose,
                        },
                    ),
                    "Lin. CFR (A)": (
                        LinearCFR,
                        n_iters,
                        {
                            "game_name": game,
                            "regret_minimizer": rm.RegretMatcherDiscounted,
                            "alternating": True,
                            "do_print": verbose,
                        },
                    ),
                    "Lin. CFR (S)": (
                        LinearCFR,
                        n_iters,
                        {
                            "game_name": game,
                            "regret_minimizer": rm.RegretMatcherDiscounted,
                            "alternating": False,
                            "do_print": verbose,
                        },
                    ),
                    "Lin. CFR+ (A)": (
                        LinearCFR,
                        n_iters,
                        {
                            "game_name": game,
                            "regret_minimizer": rm.RegretMatcherDiscountedPlus,
                            "alternating": True,
                            "do_print": verbose,
                        },
                    ),
                    "Lin. CFR+ (S)": (
                        LinearCFR,
                        n_iters,
                        {
                            "game_name": game,
                            "regret_minimizer": rm.RegretMatcherDiscountedPlus,
                            "alternating": False,
                            "do_print": verbose,
                        },
                    ),
                }.items()
            )
            augmented_jobs = []
            for alg, args in jobs:
                kwargs = args[-1]
                if kwargs.pop("stochastic_solver", False):
                    augmented_jobs.extend(
                        [
                            (alg, args[:-1] + (kwargs | {"seed": seed},))
                            for seed in rng.integers(0, int(1e6), size=stochastic_seeds)
                        ]
                    )
                else:
                    augmented_jobs.append((alg, args))
            results = pool.imap_unordered(
                main_wrapper,
                (
                    (name, args_and_kwargs[:-1], args_and_kwargs[-1])
                    for (name, args_and_kwargs) in augmented_jobs
                ),
            )
            expl_dict = defaultdict(list)

            with tqdm(
                total=len(augmented_jobs),
                desc=f"Running CFR variants in multiprocess on {n_cpu} cpus",
            ) as pbar:
                for result in results:
                    pbar.update()
                    name, values = result
                    expl_dict[name].append(values)

        with open(os.path.join(".", filename), "wb") as file:
            pickle.dump(expl_dict, file)
    else:
        with open(os.path.join(".", filename), "rb") as file:
            expl_dict = pickle.load(file)
    averaged_values = {
        name: [running_mean(values, window_size=20) for values in expl_values]
        for name, expl_values in expl_dict.items()
    }
    plot_cfr_convergence(
        n_iters,
        averaged_values,
        game_name=" ".join([s.capitalize() for s in game.split("_")]),
        save=True,
        save_name=f"{game}",
    )
