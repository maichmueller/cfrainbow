from typing import Dict, List

import matplotlib.cm
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from collections import Counter


def prep_data(iters_run: int, value_list: List[List[float]]):
    x = np.arange(1, iters_run + 1)

    iteration_counts = np.flip(
        np.sort(np.asarray([len(vals) for vals in value_list])), 0
    )
    iter_beginnings = iters_run - iteration_counts
    freq_arr = np.asarray(
        sorted(Counter(iter_beginnings).items(), key=lambda x: x[0]),
        dtype=float,
    )
    absolute_freq = np.cumsum(freq_arr, axis=0)[:, 1]
    relative_freq = absolute_freq / len(value_list)
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
            + [torch.Tensor(list(reversed(a))) for a in value_list],
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
    stderr = np.nanstd(nanpadded_iter_values_matrix, ddof=1, axis=0) / np.sqrt(
        np.isnan(nanpadded_iter_values_matrix).sum(axis=0)
    )
    nan_mask = np.isnan(mean_band)
    x_to_plot = (x - 1)[~nan_mask]
    # repackage the x, y data points as a list of 2D coordinates
    mean_to_plot = mean_band[~nan_mask]
    stderr_to_plot = stderr[~nan_mask]
    return (
        x_to_plot,
        mean_to_plot,
        stderr_to_plot,
        relative_freq[iter_to_bucket[~no_value_mask]],
        relative_freq_per_iter,
    )


def plot(
    iters_run: int,
    method_to_data_dict: Dict[str, List[List[float]]],
    save_name: str,
    linewidth=1,
):
    with plt.style.context("bmh"):
        fig, ax = plt.subplots()
        cmap = matplotlib.cm.get_cmap("tab20")

        manual_legend_handles = []

        for i, (method, list_of_values) in enumerate(method_to_data_dict.items()):

            linestyle = "--" if "(A)" in method else "-"

            color = cmap(i)

            if len(list_of_values) == 1:
                iter_offset = iters_run - len(list_of_values[0])
                ax.plot(
                    np.arange(iter_offset, iters_run)
                    + 1,  # +1 to plot from iteration 1 on the scale, not 0
                    list_of_values[0],
                    label=method,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    color=color,
                )
            else:
                (
                    x_to_plot,
                    mean_to_plot,
                    stderr_to_plot,
                    relative_freq_existing_values,
                    relative_freq_for_all_iter,
                ) = prep_data(iters_run, list_of_values)

                points = np.array([x_to_plot, mean_to_plot]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                lc = LineCollection(
                    segments,
                    linewidths=relative_freq_for_all_iter * linewidth,
                    color=color,
                    alpha=relative_freq_for_all_iter,
                )
                ax.add_collection(lc)
                # add the legend entry manually
                manual_legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        label=method,
                        linewidth=linewidth,
                        linestyle=linestyle,
                        color=color,
                    )
                )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Log scale metric")
        ax.set_yscale("log", base=10)
        ax.set_title(f"My Title")
        manual_legend_handles.extend(ax.get_legend_handles_labels()[0])
        ax.legend(
            handles=manual_legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            fancybox=True,
            shadow=True,
            ncol=4,
        )
        plt.savefig(f"{save_name}_iters_{iters_run}.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    n_iters = 1000

    rng = np.random.default_rng(0)
    values_dict = {
        "stochastic (A)": [
            rng.normal(100, variance, size=n_iters - offset)
            / np.arange(offset, n_iters)
            for variance, offset in zip(
                rng.integers(0.5, 5, size=20), (rng.integers(5, 300, size=20))
            )
        ],
        "stochastic (S)": [
            rng.normal(100, variance, size=n_iters - offset)
            / (np.arange(offset, n_iters) ** 1.5)
            for variance, offset in zip(
                rng.integers(0.5, 5, size=20), (rng.integers(5, 300, size=20))
            )
        ],
        "deterministic (A)": [
            rng.normal(100, 1, size=n_iters - 2) / (np.arange(2, n_iters) ** 2)
        ],
        "deterministic (S)": [
            rng.normal(100, 1, size=n_iters - 2) / (np.arange(2, n_iters) ** 2.5)
        ],
    }

    plot(
        n_iters,
        values_dict,
        save_name=f"minimal_working_example",
    )
