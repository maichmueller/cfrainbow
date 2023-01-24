import itertools
import operator
from collections import namedtuple, defaultdict
from functools import reduce
from typing import Sequence, Dict, Tuple, Set, List

import numpy as np
import pyspiel

from type_aliases import (
    Infostate,
    Action,
    NormalFormPlan,
    NormalFormStrategySpace,
    SequenceFormPlan,
)
from utils import all_states_gen


_informed_sequence = namedtuple("_informed_sequence", "infostate action depth")


def _informed_seq_filter(x: _informed_sequence):
    return x.depth * 1e8 + sum(ord(c) for c in x.infostate)


class InformedActionList:
    def __init__(self, infostate: Infostate, actions: Sequence[Action], depth: int):
        self.infostate: Infostate = infostate
        self.actions: Sequence[Action] = actions
        self.depth: int = depth

    def __iter__(self):
        return iter(
            _informed_sequence(self.infostate, action, self.depth)
            for action in self.actions
        )

    def __repr__(self):
        return f"{self.infostate}, {self.actions}, {self.depth}"


def normal_form_strategy_space(
    game: pyspiel.Game, *players: int
) -> Dict[int, Set[NormalFormPlan]]:
    if not players:
        players: List[int] = list(range(game.num_players()))
    action_spaces = {p: [] for p in players}
    strategy_spaces = {}
    seen_infostates = set()
    for state, depth in all_states_gen(root=game.new_initial_state()):
        player = state.current_player()
        if player in players:
            infostate = state.information_state_string(player)
            if infostate not in seen_infostates:
                action_spaces[player].append(
                    InformedActionList(infostate, state.legal_actions(), depth)
                )
                seen_infostates.add(infostate)

    for player, action_space in action_spaces.items():
        strategy_spaces[player] = set(
            tuple(
                (sorted_plan.infostate, sorted_plan.action)
                for sorted_plan in sorted(plan, key=lambda x: x.depth)
            )
            for plan in itertools.product(
                *sorted(
                    action_space,
                    key=_informed_seq_filter,
                )
            )
        )
    return strategy_spaces


def sequence_space(
    game: pyspiel.Game, *players: int
) -> Dict[int, Dict[Tuple[Infostate, Action], Tuple[Tuple[Action], Set[Infostate]]]]:
    if not players:
        players = list(range(game.num_players()))
    spaces: Dict[
        int, Dict[Tuple[Infostate, Action], Tuple[Tuple[Action], Set[Infostate]]]
    ] = {}

    for player in players:
        sequences: Dict[Tuple[Infostate, Action], Tuple[Action]] = dict()
        sequence_succ_set: Dict[Tuple[Infostate, Action], Set[Infostate]] = defaultdict(
            set
        )
        root = game.new_initial_state()
        stack = [(root.clone(), [], 0)]
        while stack:
            state, plan, depth = stack.pop()
            if state.is_terminal():
                n_seqs = len(plan)
                action_seq: List[Action] = []
                for i, seq in enumerate(plan):
                    action_seq.append(seq.action)
                    key = (seq.infostate, seq.action)
                    if i < n_seqs - 1:
                        sequence_succ_set[key].add(plan[i + 1].infostate)
                    sequences[key] = tuple(action_seq)
            elif state.is_chance_node():
                stack.extend(
                    [
                        (state.child(action), plan.copy(), depth + 1)
                        for action, _ in state.chance_outcomes()
                    ]
                )
            else:
                curr_player = state.current_player()
                if curr_player != player:
                    stack.extend(
                        [
                            (state.child(action), plan.copy(), depth + 1)
                            for action in state.legal_actions()
                        ]
                    )
                else:
                    infostate = state.information_state_string(player)
                    actions = state.legal_actions()
                    # reserve more memory for the list
                    actual_stack_length = len(stack)
                    stack.extend([None] * len(actions))
                    for i, action in enumerate(actions):
                        plan_copy = plan.copy()
                        plan_copy.append(
                            _informed_sequence(infostate, action, depth + 1)
                        )
                        stack[actual_stack_length + i] = (
                            state.child(action),
                            plan_copy,
                            depth + 1,
                        )
        spaces[player] = {
            seq: (action_list, sequence_succ_set[seq])
            for seq, action_list in sequences.items()
        }
    return spaces


def reduced_normal_form_strategy_space(
    game: pyspiel.Game, *players: int
) -> Dict[int, Set[NormalFormPlan]]:
    if not players:
        players = list(range(game.num_players()))
    spaces = {}
    sequences_per_player = sequence_space(game, *players)
    for player, sequences in sequences_per_player.items():
        # root sequences are those for which the action list leading up the infostate I and including the action to
        # take at I are exactly 1 (only the action to take at I)
        root_sequences = list(filter(lambda x: len(x[1][0]) == 1, sequences.items()))
        available_actions = defaultdict(list)
        for (infostate, action), action_list in sequences.items():
            available_actions[infostate].append(action)

        root_infostates = set(infostate for (infostate, _), _ in root_sequences)

        infostate_sequences = {s: [] for s in root_infostates}
        for root_infostate, seq_list in infostate_sequences.items():
            stack = [(root_infostate, [], 0)]
            while stack:
                infostate, seq, depth = stack.pop()
                for action in available_actions[infostate]:
                    seq_copy = seq.copy()
                    seq_copy.append(_informed_sequence(infostate, action, depth))

                    if succ_infostates := sequences[(infostate, action)][1]:
                        stack.extend(
                            [
                                (succ_infostate, seq_copy, depth + 1)
                                for succ_infostate in succ_infostates
                            ]
                        )
                    else:
                        seq_list.append(seq_copy)

        final_plans = [None] * reduce(
            operator.mul, (len(seq) for seq in infostate_sequences.values())
        )
        for i, seq_lists_combo in enumerate(
            itertools.product(*infostate_sequences.values())
        ):
            final_plans[i] = tuple(
                (infostate, action)
                for infostate, action, _ in sorted(
                    itertools.chain(*seq_lists_combo), key=_informed_seq_filter
                )
            )
        spaces[player] = set(final_plans)
    return spaces


def normal_form_expected_payoff(game: pyspiel.Game, *joint_plan: NormalFormPlan):
    joint_plan_dict = {
        infostate: action for infostate, action in itertools.chain(*joint_plan)
    }
    root = game.new_initial_state()
    stack = [(root, 1.0)]
    expected_payoff = np.zeros(root.num_players())
    while stack:
        s, chance = stack.pop()
        if s.is_chance_node():
            for outcome, prob in s.chance_outcomes():
                stack.append((s.child(outcome), chance * prob))
        elif s.is_terminal():
            expected_payoff += chance * np.asarray(s.returns())
        else:
            infostate = s.information_state_string(s.current_player())
            if infostate not in joint_plan_dict:
                # the infostate not being part of the plan means this is a reduced plan and the current point is
                # unreachable by the player due to previous decisions. The expected outcome of this is hence simply 0,
                # since it is never reached.
                continue
            s.apply_action(joint_plan_dict[infostate])
            stack.append((s, chance))
    return expected_payoff


def normal_form_expected_payoff_table(
    game: pyspiel.Game, strategy_spaces: Sequence[NormalFormStrategySpace]
):
    payoffs: Dict[Tuple[NormalFormPlan], Sequence[float]] = dict()
    for joint_profile in itertools.product(*strategy_spaces):
        payoffs[joint_profile] = normal_form_expected_payoff(game, *joint_profile)
    return payoffs
