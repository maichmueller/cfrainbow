from copy import deepcopy
from enum import Enum
from typing import Dict, Mapping

import pyspiel

from cfr_discounted import DiscountedCFR
from utils import (
    print_final_policy_profile,
    print_policy_profile,
    to_pyspiel_tab_policy,
)


Action = int
Probability = float
Regret = float
Value = float
Infostate = str


class Players(Enum):
    chance = -1
    player1 = 0
    player2 = 1


class LinearCFR(DiscountedCFR):
    def __init__(
        self,
        root_state: pyspiel.State,
        curr_policy_list: list[Dict[Infostate, Mapping[Action, Probability]]],
        average_policy_list: list[Dict[Infostate, Mapping[Action, Probability]]],
        *,
        simultaneous_updates: bool = True,
        do_regret_matching_plus: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            root_state,
            curr_policy_list,
            average_policy_list,
            simultaneous_updates=simultaneous_updates,
            do_regret_matching_plus=do_regret_matching_plus,
            verbose=verbose,
            alpha=1.0,
            beta=1.0,
            gamma=1.0,
        )


def main(
    n_iter,
    simultaneous_updates: bool = True,
    rm_plus: bool = False,
    do_print: bool = True,
):
    from open_spiel.python.algorithms import exploitability

    if do_print:
        print(
            f"Running Discounted CFR with"
            f" {'simultaneous updates' if simultaneous_updates else 'alternating updates'}"
            f" for {n_iter} iterations."
        )

    expl_values = []
    game = pyspiel.load_game("kuhn_poker")
    root_state = game.new_initial_state()
    n_players = list(range(root_state.num_players()))
    current_policies = [{} for _ in n_players]
    average_policies = [{} for _ in n_players]
    solver = LinearCFR(
        root_state,
        current_policies,
        average_policies,
        simultaneous_updates=simultaneous_updates,
        do_regret_matching_plus=rm_plus,
        verbose=do_print,
    )
    for i in range(n_iter):
        solver.iterate()

        if simultaneous_updates or (not simultaneous_updates and i > 1):
            expl_values.append(
                exploitability.exploitability(
                    game, to_pyspiel_tab_policy(average_policies),
                )
            )

            if do_print:
                print(
                    f"-------------------------------------------------------------"
                    f"--> Exploitability {expl_values[-1]: .5f}"
                )
                print_policy_profile(deepcopy(average_policies))
                print(
                    f"---------------------------------------------------------------"
                )
    if do_print:
        print_final_policy_profile(average_policies)

    return expl_values


if __name__ == "__main__":
    main(n_iter=2000, do_print=False)
