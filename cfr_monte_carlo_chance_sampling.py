from copy import deepcopy
from enum import Enum
from typing import Optional, Union

import numpy as np

from cfr import CFR
import pyspiel
from open_spiel.python.algorithms import exploitability

from rm import all_states_gen
from utils import (
    print_final_policy_profile,
    print_policy_profile,
    to_pyspiel_tab_policy,
)


class ChanceSamplingCFR(CFR):
    def __init__(
        self, *args, seed: Optional[Union[int, np.random.Generator]] = None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.rng: np.random.Generator = np.random.default_rng(seed)

    def _traverse_chance_node(self, state, reach_prob, updating_player, action_values):
        outcome, outcome_prob = self.rng.choice(state.chance_outcomes())
        state.apply_action(int(outcome))
        return self._traverse(state, reach_prob, updating_player)


def main(n_iter, simultaneous_updates: bool = True, do_print: bool = True):

    if do_print:
        print(
            f"Running CFR with "
            f"{'simultaneous updates' if simultaneous_updates else 'alternating updates'} "
            f"for {n_iter} iterations."
        )

    expl_values = []
    game = pyspiel.load_game("kuhn_poker")
    root_state = game.new_initial_state()
    n_players = list(range(root_state.num_players()))
    current_policies = [{} for _ in n_players]
    average_policies = [{} for _ in n_players]
    all_infostates = {
        state.information_state_string(state.current_player())
        for state in all_states_gen(game=game)
    }
    solver = ChanceSamplingCFR(
        root_state,
        current_policies,
        average_policies,
        simultaneous_updates=simultaneous_updates,
        verbose=do_print,
    )

    for i in range(n_iter):
        solver.iterate()

        if sum(map(lambda p: len(p), solver.average_policy())) == len(
            all_infostates
        ) and (simultaneous_updates or (not simultaneous_updates and i > 1)):
            expl_values.append(
                exploitability.exploitability(
                    game,
                    to_pyspiel_tab_policy(average_policies),
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
    main(n_iter=2000, simultaneous_updates=False, do_print=True)
