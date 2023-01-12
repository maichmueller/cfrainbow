from collections import defaultdict
from copy import deepcopy
from typing import Optional, Dict, Sequence, Union

from open_spiel.python.algorithms import exploitability

from rm import (
    regret_matching,
    all_states_gen,
    counterfactual_reach_prob,
)
import pyspiel
import numpy as np

from utils import (
    to_pyspiel_tab_policy,
    sample_on_policy,
    print_final_policy_profile,
    print_policy_profile,
)


Action = int
Probability = float
Regret = float
Value = float
Infostate = str


class PureCFR:
    def __init__(
        self,
        root_state: pyspiel.State,
        curr_policy_list: list[Dict[str, Dict[int, float]]],
        average_policy_list: list[Dict[str, Dict[int, float]]],
        *,
        seed: Optional[Union[int, np.random.Generator]] = None,
        simultaneous_updates: bool = True,
        verbose: bool = False,
    ):
        self.root_state = root_state
        self.n_players = list(range(root_state.num_players()))
        self.regret_table: list[Dict[str, Dict[int, float]]] = [
            {} for p in self.n_players
        ]
        self.curr_policy = curr_policy_list
        self.avg_policy = average_policy_list
        self.iteration = 0
        self.plan = {}
        self.rng = np.random.default_rng(seed)

        self._simultaneous_updates = simultaneous_updates
        self._verbose = verbose

    def _get_average_strategy(
        self, current_player, infostate, actions: Optional[Sequence[int]] = None
    ):
        if infostate not in (player_policy := self.avg_policy[current_player]):
            player_policy[infostate] = {a: 0.0 for a in actions}
        return player_policy[infostate]

    def _get_information_state(self, current_player, state):
        infostate = state.information_state_string(current_player)
        if infostate not in (player_policy := self.curr_policy[current_player]):
            las = state.legal_actions()
            player_policy[infostate] = {action: 1 / len(las) for action in las}
        return infostate

    def _get_regret_table(self, current_player: int, infostate: str):
        if infostate not in self.regret_table[current_player]:
            self.regret_table[current_player][infostate] = defaultdict(float)

        return self.regret_table[current_player][infostate]

    def iterate(
        self, updating_player: Optional[int] = None,
    ):
        if self._verbose:
            print(
                "\nIteration",
                self.iteration
                if self._simultaneous_updates
                else f"{self.iteration // 2} {(self.iteration % 2 + 1)}/2",
            )

        if updating_player is None and not self._simultaneous_updates:
            updating_player = self.iteration % 2

        root_reach_probabilities = (
            {player: 1.0 for player in self.n_players}
            if self._simultaneous_updates
            else None
        )

        self.plan.clear()
        values = self._traverse(
            self.root_state.clone(), updating_player, root_reach_probabilities
        )
        self._apply_regret_matching(updating_player)
        self.iteration += 1
        return values

    def _traverse(
        self,
        state: pyspiel.State,
        updating_player: Optional[int] = None,
        reach_prob: Optional[Dict[int, Probability]] = None,
    ):
        if state.is_terminal():
            return state.returns()

        current_player = state.current_player()
        action_values = {}

        if state.is_chance_node():
            outcomes_probs = state.chance_outcomes()
            outcome = sample_on_policy(
                values=[outcome[0] for outcome in outcomes_probs],
                policy=[outcome[1] for outcome in outcomes_probs],
                rng=self.rng,
            )
            return self._traverse(state.child(int(outcome)), updating_player, reach_prob)

        infostate = self._get_information_state(current_player, state)
        actions = state.legal_actions()
        player_policy = self._get_current_strategy(current_player, infostate, actions)

        sampled_action = self._get_sampled_action(infostate, player_policy)

        if self._simultaneous_updates or current_player == updating_player:
            child_reach_prob = reach_prob

            for action in actions:

                if self._simultaneous_updates:
                    child_reach_prob = deepcopy(reach_prob)
                    child_reach_prob[current_player] *= player_policy[action]

                action_values[action] = self._traverse(
                    state.child(action), updating_player, child_reach_prob
                )

            state_value = action_values[sampled_action]

            regrets = self._get_regret_table(current_player, infostate)

            # TODO: Try to understand this! Simultaneous updating PURE CFR only works if we actually multiply the
            #  counterfactual reach probability to the regret increment. Therefore, I am mixing the regret update rule
            #  from chance-sampling and pure cfr: The state value is computed according to pure-car's sampled
            #  action-value, but the difference of each action value to the state value is then multiplied by the
            #  cf. reach probability as in chance-sampling. Why this ends up being a correct regret update is unclear,
            #  even more so because the strategy update is exactly according to pure cfr, and not chance-sampling.
            prob_weight = (
                counterfactual_reach_prob(reach_prob, current_player)
                if self._simultaneous_updates
                else 1.0
            )
            for action, action_vs in action_values.items():
                regrets[action] += prob_weight * (
                    action_vs[current_player] - state_value[current_player]
                )

            if self._simultaneous_updates:
                self._get_average_strategy(current_player, infostate, actions)[
                    sampled_action
                ] += 1

        else:
            self._get_average_strategy(current_player, infostate, actions)[
                sampled_action
            ] += 1
            state.apply_action(sampled_action)
            state_value = self._traverse(state, updating_player)

        return state_value

    def _get_current_strategy(self, current_player, infostate, actions):
        if infostate not in (player_policy := self.curr_policy[current_player]):
            player_policy[infostate] = {
                action: 1.0 / len(actions) for action in actions
            }
        return player_policy[infostate]

    def _get_sampled_action(self, infostate, player_policy):
        if infostate not in self.plan:
            actions = list(player_policy.keys())
            self.plan[infostate] = self.rng.choice(
                actions, p=[player_policy[action] for action in actions],
            )
        sampled_action = self.plan[infostate]
        return sampled_action

    def _apply_regret_matching(self, updating_player: Optional[int] = None):
        for player, player_policy in enumerate(self.curr_policy):
            if self._simultaneous_updates or player == updating_player:
                for infostate, regret_dict in self.regret_table[player].items():
                    regret_matching(player_policy[infostate], regret_dict)

    def average_policy(self, player: Optional[int] = None):
        if player is None:
            return self.avg_policy
        else:
            return [self.avg_policy[player]]


def main(n_iter, simultaneous_updates: bool = True, do_print: bool = True):

    if do_print:
        print(
            f"Running Pure CFR with "
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
    solver = PureCFR(
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
            average_policy = solver.average_policy()
            expl_values.append(
                exploitability.exploitability(
                    game, to_pyspiel_tab_policy(average_policy),
                )
            )

            if do_print:
                print(
                    f"-------------------------------------------------------------"
                    f"--> Exploitability {expl_values[-1]: .5f}"
                )
                print_policy_profile(deepcopy(average_policy))
                print(
                    f"---------------------------------------------------------------"
                )
    if do_print:
        print_final_policy_profile(solver.average_policy())

    return expl_values


if __name__ == "__main__":
    main(n_iter=20000, simultaneous_updates=True, do_print=True)
    # main(n_iter=20000, simultaneous_updates=False, do_print=True)
