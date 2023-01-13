from typing import Mapping
import pandas as pd
import numpy as np
from itertools import product

Lisa = 0
Bart = 1
Player = int
Action = str
Regret = float
Probability = float

action_set = ["Rock", "Paper", "Scissors"]

payoff_lisa = pd.DataFrame(
    np.empty((3, 3), dtype=int), index=action_set, columns=action_set
)
payoff_lisa.at["R", "R"] = 0
payoff_lisa.at["R", "P"] = -1
payoff_lisa.at["R", "S"] = 1
payoff_lisa.at["P", "R"] = 1
payoff_lisa.at["P", "P"] = 0
payoff_lisa.at["P", "S"] = -1
payoff_lisa.at["S", "R"] = -1
payoff_lisa.at["S", "P"] = 1
payoff_lisa.at["S", "S"] = 0

payoff_matrix = {Lisa: payoff_lisa, Bart: -1 * payoff_lisa}

current_strategies = {
    Lisa: pd.Series({"Rock": 0.0, "Paper": 0.0, "Scissors": 1.0}),
    Bart: pd.Series({"Rock": 1.0, "Paper": 0.0, "Scissors": 0.0}),
}


def regret_matching(cumulative_regret: Mapping[Action, Regret]):
    pos_regrets = {
        action: max(0, regret) for action, regret in cumulative_regret.items()
    }
    if any([regret > 0 for regret in pos_regrets.values()]):
        new_policy = {
            action: regret / sum(pos_regrets.values())
            for action, regret in pos_regrets.items()
        }
    else:
        uniform_distribution = {
            action: 1.0 / len(cumulative_regret) for action in cumulative_regret.keys()
        }
        new_policy = uniform_distribution

    return pd.Series(new_policy, index=action_set)


def u(
    p: Player,
    strategy_lisa: Mapping[Action, Probability],
    strategy_bart: Mapping[Action, Probability],
):
    payoff = 0.0
    # compute the expected payoff for the player given the two strategies
    for action_lisa, action_bart in product(action_set, action_set):
        action_initials = action_lisa[0], action_bart[0]
        payoff += (
            strategy_lisa[action_lisa]    # likelihood that Lisa played her action
            * strategy_bart[action_bart]  # likelihood that Bart played his action
            * payoff_matrix[p].at[action_initials]  # the payoff for the chosen player for this action combination
        )
    return payoff


def current_regret(
    deviation_action: Action,
    player: Player,
    strategy_lisa: Mapping[Action, Probability],
    strategy_bart: Mapping[Action, Probability],
):
    deviating_strategy = pd.Series({act: 0.0 for act in action_set})
    deviating_strategy[deviation_action] = 1.0

    if player == Lisa:
        deviation_payoff = u(player, deviating_strategy, strategy_bart)
    else:
        deviation_payoff = u(player, strategy_lisa, deviating_strategy)

    expected_payoff = u(player, strategy_lisa, strategy_bart)

    return deviation_payoff - expected_payoff


def main(bart_learns: bool = False, horizon: int = 5):
    rng = np.random.default_rng(0)

    cumulative_regret_table = {
        Lisa: pd.Series(np.zeros(len(action_set)), index=action_set),
        Bart: pd.Series(np.zeros(len(action_set)), index=action_set),
    }

    cumulative_strategies_table = {
        Lisa: current_strategies[Lisa].copy(),
        Bart: current_strategies[Bart].copy(),
    }

    for round in range(1, horizon + 1):
        # lisa chooses action by sampling from her strategy
        action_lisa = rng.choice(action_set, p=current_strategies[Lisa])
        action_bart = rng.choice(action_set, p=current_strategies[Bart])

        print("\n ROUND ", round, "\n")
        print("Action Lisa:", action_lisa)
        print("Action Bart:", action_bart)

        # we compute the current regret and append it to the cumulative one
        for action in action_set:
            # add the current round's regret to the cumulative regret
            cumulative_regret_table[Lisa][action] += current_regret(
                action, Lisa, current_strategies[Lisa], current_strategies[Bart]
            )
            if bart_learns:
                cumulative_regret_table[Bart][action] += current_regret(
                    action, Bart, current_strategies[Lisa], current_strategies[Bart]
                )

        # compute the new strategies from the cumulative regret
        current_strategies[Lisa] = regret_matching(cumulative_regret_table[Lisa])
        # add the current strategies to the running sum of past strategies
        cumulative_strategies_table[Lisa] += current_strategies[Lisa]
        
        if bart_learns:
            current_strategies[Bart] = regret_matching(cumulative_regret_table[Bart])
            cumulative_strategies_table[Bart] += current_strategies[Bart]

        print_status(
            cumulative_regret_table,
            current_strategies,
            {
                player: table / (round + 1)
                for player, table in cumulative_strategies_table.items()
            },
        )


def print_status(cumul_regret, current_strategies, avg_strategies):
    dfs = [None, None]
    for player in [Lisa, Bart]:
        dfs[player] = pd.concat(
            [
                cumul_regret[player].to_frame(),
                current_strategies[player].to_frame(),
                avg_strategies[player].to_frame(),
            ],
            axis=1,
        )
        dfs[player].columns = ["Cumul. Regret", "Curr. Strategy", "Avg. Strategy"]
    output_lisa = dfs[Lisa].to_string().split("\n")
    output_bart = dfs[Bart].to_string(index=False).split("\n")

    longest_action_filler = " " * (max([len(a) for a in action_set]) + 2)
    out_lines = [
        " " * output_lisa[0].find(". Strategy")
        + "Lisa"
        + " " * (len(output_lisa[0]) - output_lisa[0].find(". Strategy") - 4)
        + longest_action_filler
        + " " * output_bart[0].find(". Strategy")
        + "Bart"
    ]
    for line_l, line_b in zip(output_lisa, output_bart):
        out_lines.append(line_l + longest_action_filler + line_b)
    print("\n".join(out_lines))


if __name__ == "__main__":
    main(bart_learns=True, horizon=50000)
