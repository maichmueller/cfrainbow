import operator
import os
import pickle
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional

import matplotlib.cm
import matplotlib.font_manager
import matplotlib.patheffects as pe
import numpy as np
import torch
from matplotlib import ft2font
from matplotlib import pyplot as plt
from tqdm import tqdm

from cfrainbow import rm
from cfrainbow.cfr import *
from cfrainbow.main import run


def plot_rainbow(
    iters_run: int,
    map_of_algorithm_to_expl_lists: Dict[str, List[List[float]]],
    save: bool = False,
    save_name: Optional[str] = None,
    light: bool = False,
):
    dir_path = os.path.split(os.path.abspath(__file__))[0]
    font_files = matplotlib.font_manager.findSystemFonts(
        fontpaths=os.path.join(dir_path, "font"), fontext="ttf"
    )
    font_files += matplotlib.font_manager.findSystemFonts(
        fontpaths=os.path.join(dir_path, "font"), fontext="otf"
    )
    font_names = set()
    for font_file in font_files:
        matplotlib.font_manager.fontManager.addfont(font_file)
        font = ft2font.FT2Font(font_file)
        font_names.add(matplotlib.font_manager.ttfFontProperty(font).name)

    style_list = ["seaborn-v0_8-dark"]
    for style in style_list:
        for font in font_names:
            font = os.path.splitext(os.path.split(font)[-1])[0]
            with plt.style.context(style):
                plt.figure()
                cmap = matplotlib.cm.get_cmap("gist_rainbow")
                color_linspace = np.linspace(0, 1, len(map_of_algorithm_to_expl_lists))

                def get_color(index):
                    return cmap(
                        color_linspace[len(map_of_algorithm_to_expl_lists) - 1 - index]
                    )

                list_of_algorithm_and_expl_lists = sorted(
                    list(map_of_algorithm_to_expl_lists.items()),
                    key=operator.itemgetter(0),
                )
                linewidth = 0.5
                x = np.arange(1, iters_run + 1)
                fig, ax = plt.subplots(figsize=(11.5, 3.8))

                sorting = [
                    (i, expl_list[0][-1])
                    for i, (algo, expl_list) in enumerate(
                        list_of_algorithm_and_expl_lists
                    )
                ]
                sorting = sorted(sorting, key=operator.itemgetter(1), reverse=True)
                sorted_algorithm_to_expl_list = [
                    list_of_algorithm_and_expl_lists[i[0]] for i in sorting
                ]

                for i, (algo, expl_list) in enumerate(sorted_algorithm_to_expl_list):
                    color = get_color(i)
                    ax.plot(
                        x[iters_run - len(expl_list[0]) :],
                        expl_list[0],
                        label=algo,
                        linewidth=linewidth,
                        color=color,
                    )

                packed = (
                    torch.nn.utils.rnn.pad_sequence(
                        [  # full-length zeros tensor to force padding for the entire run_iters length
                            torch.zeros(iters_run)
                        ]
                        + [
                            torch.Tensor(list(reversed(l[0])))
                            for _, l in sorted_algorithm_to_expl_list
                        ],
                        batch_first=True,
                    )
                    .flip(dims=(1,))
                    .numpy()
                )
                packed = np.where(packed == 0, np.max(packed), packed)
                sorted_packed = np.empty_like(packed).T
                for i, column in enumerate(packed.T):
                    sorted_packed[i, :] = np.sort(column)[::-1]
                sorted_packed = sorted_packed.T[1:]
                for i, ith_row in enumerate(sorted_packed[:-1]):
                    color = get_color(i)
                    ith_p1_row = sorted_packed[i + 1, :]
                    ax.fill_between(
                        x,
                        ith_p1_row,
                        ith_row,
                        color=color,
                        alpha=0.3,
                    )

                ax.set_yscale("log", base=10)

                def get_color(s):
                    return cmap(np.linspace(0, 1, len("ainbow"))["ainbow".index(s)])

                if light:
                    bg_color = "white"
                    text_color = "black"
                else:
                    bg_color = "black"
                    text_color = "white"

                transparent_bg = True

                title_fontsize = 90
                title_tinyfontsize = 45

                title_x_start = 0.655
                title_y_start = 0.865
                letter_x_offset = [0.015] * len("ainbow")
                letter_x_offset[2] = 0.0121
                letter_x_offset[3] = 0.0147
                letter_y_offset = 0.0011
                rotation = -3
                ax.text(
                    title_x_start,
                    title_y_start,
                    "CFR",
                    fontsize=title_fontsize,
                    rotation=rotation,
                    fontfamily=font,
                    transform=ax.transAxes,
                    weight="bold",
                    color=text_color,
                )
                for i, letter in enumerate("ainbow"):
                    ax.text(
                        title_x_start + 0.195 + i * letter_x_offset[i],
                        title_y_start - 0.064 - i * letter_y_offset,
                        letter,
                        fontfamily=font,
                        rotation=rotation,
                        fontsize=title_tinyfontsize,
                        transform=ax.transAxes,
                        color=get_color(letter),
                        weight="bold",
                        path_effects=[pe.withStroke(linewidth=0.3, foreground="black")],
                    )

                label_fontsize = 25
                expl_loc = (
                    0.0,
                    0.67,
                )
                ax.text(
                    *expl_loc,
                    "Exploitability",
                    fontsize=label_fontsize,
                    rotation=90,
                    fontfamily=font,
                    transform=ax.transAxes,
                    weight="bold",
                    color=text_color,
                )
                plt.arrow(
                    expl_loc[0] + 0.02,
                    expl_loc[1] - 0.065,
                    0,
                    -0.3,
                    width=0.00001,
                    length_includes_head=True,
                    head_width=0.005,
                    head_length=0.01,
                    shape="right",
                    transform=ax.transAxes,
                    color=text_color,
                )
                round_loc = (
                    0.05,
                    0.03,
                )
                ax.text(
                    *round_loc,
                    "Round",
                    fontsize=label_fontsize,
                    fontfamily=font,
                    transform=ax.transAxes,
                    weight="bold",
                    color=text_color,
                )
                plt.arrow(
                    round_loc[0] + 0.07,
                    round_loc[1] + 0.01,
                    0.15,
                    0,
                    width=0.00001,
                    length_includes_head=True,
                    head_width=0.015,
                    head_length=0.0055,
                    shape="right",
                    transform=ax.transAxes,
                    color=text_color,
                )
                ax.set_facecolor(bg_color)
                # ax.tick_params(axis="both", which="major", labelsize=5)
                # ax.tick_params(axis="both", which="minor", labelsize=0)
                #
                # ax.tick_params(axis="y", direction="in", pad=-17, colors=text_color)
                # ax.tick_params(axis="x", direction="in", pad=-10, colors=text_color)

                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                legend = ax.legend(
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.02),
                    fancybox=True,
                    ncol=5,
                    prop=dict(family=font, size=14),
                )
                legend.get_frame().set_facecolor(bg_color)
                for text in legend.get_texts():
                    text.set_color(text_color)

                if not save:
                    plt.show(bbox_inches="tight")
                else:
                    plt.savefig(
                        f"cfrainbow_readme_banner_{'light' if light else 'dark'}_{save_name}.png",
                        dpi=300,
                        bbox_inches="tight",
                        transparent=transparent_bg,
                    )
                plt.close(fig)


def running_mean(values, window_size: int = 10):
    return [
        np.mean(values[max(0, -window_size + i) : i + 1]) for i in range(len(values))
    ]


def run_wrapper(args):
    name, pos_args, kwargs = args
    return name, run(*pos_args, **kwargs)


if __name__ == "__main__":
    n_iters = 1000
    verbose = False
    game = "kuhn_poker"
    # game = "leduc_poker"
    for seed in range(7, 8):
        rng = np.random.default_rng(seed)
        stochastic_seeds = 1
        n_cpu = cpu_count()

        filename = f"data_seed_{seed}.pkl"
        if not os.path.exists(os.path.join("..", filename)):
            with Pool(processes=n_cpu) as pool:
                jobs = list(
                    {
                        "Vanilla CFR": (
                            VanillaCFR,
                            n_iters,
                            {
                                "game_name": game,
                                "regret_minimizer": rm.RegretMatcher,
                                "alternating": True,
                                "do_print": verbose,
                            },
                        ),
                        "Pure CFR": (
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
                        "OS-MCCFR": (
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
                        "CS-MCCFR": (
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
                        "ES-MCCFR": (
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
                        "CFR+": (
                            PlusCFR,
                            n_iters,
                            {
                                "game_name": game,
                                "regret_minimizer": rm.RegretMatcherPlus,
                                "do_print": verbose,
                            },
                        ),
                        "Disc. CFR": (
                            DiscountedCFR,
                            n_iters,
                            {
                                "game_name": game,
                                "regret_minimizer": rm.RegretMatcherDiscounted,
                                "alternating": True,
                                "do_print": verbose,
                            },
                        ),
                        "Exp. CFR": (
                            ExponentialCFR,
                            n_iters,
                            {
                                "game_name": game,
                                "regret_minimizer": rm.RegretMatcher,
                                "alternating": True,
                                "do_print": verbose,
                            },
                        ),
                        "Lin. CFR": (
                            LinearCFR,
                            n_iters,
                            {
                                "game_name": game,
                                "regret_minimizer": rm.RegretMatcherDiscounted,
                                "alternating": True,
                                "do_print": verbose,
                            },
                        ),
                        "PCFR+": (
                            PredictivePlusCFR,
                            n_iters,
                            {
                                "game_name": game,
                                "regret_minimizer": rm.AutoPredictiveRegretMatcherPlus,
                                "alternating": True,
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
                                for seed in rng.integers(
                                    0, int(1e6), size=stochastic_seeds
                                )
                            ]
                        )
                    else:
                        augmented_jobs.append((alg, args))
                results = pool.imap_unordered(
                    run_wrapper,
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

            with open(os.path.join("..", filename), "wb") as file:
                pickle.dump(expl_dict, file)
        else:
            with open(os.path.join("..", filename), "rb") as file:
                expl_dict = pickle.load(file)
        averaged_values = {
            name: [running_mean(values, window_size=200) for values in expl_values]
            for name, expl_values in expl_dict.items()
        }
        for light in [True, False]:
            plot_rainbow(
                n_iters, averaged_values, save=True, save_name=f"{seed}", light=light
            )
