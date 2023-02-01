import itertools
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import (
    Sequence,
    Dict,
    Tuple,
    Set,
    List,
    NamedTuple,
    Union,
    Mapping,
    Optional,
)
from rich import print
import numpy as np
import pyspiel
from tqdm import tqdm

from type_aliases import (
    Infostate,
    Action,
    NormalFormPlan,
    NormalFormStrategySpace,
    Probability,
    Player,
)
from utils import all_states_gen


class DepthInformedSequence(NamedTuple):
    infostate: Infostate
    action: Action
    depth: int


@dataclass
class SequenceAttr:
    action_seq: Tuple[Action]
    succ_infostates: Set[Infostate]


def depth_informed_seq_filter(x: DepthInformedSequence):
    # sort by depth first, then by the infostate str comparison
    # return x.depth, x.infostate
    # sort by length of infostate first, then by the alphabetical order of the infostate str
    return len(x.infostate), x.infostate


def infostate_filter(istate: Infostate):
    return len(istate), istate


class InformedActionList:
    def __init__(self, infostate: Infostate, actions: Sequence[Action], depth: int):
        self.infostate: Infostate = infostate
        self.actions: Sequence[Action] = actions
        self.depth: int = depth

    def __iter__(self):
        return iter(
            DepthInformedSequence(self.infostate, action, self.depth)
            for action in self.actions
        )

    def __repr__(self):
        return f"{self.infostate}, {self.actions}, {self.depth}"


def normal_form_strategy_space(
    game: pyspiel.Game, *players: Player
) -> Dict[Player, Set[NormalFormPlan]]:
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
                for sorted_plan in sorted(plan, key=depth_informed_seq_filter)
            )
            for plan in itertools.product(
                *sorted(
                    action_space,
                    key=depth_informed_seq_filter,
                )
            )
        )
    return strategy_spaces


def sequence_space(
    game: pyspiel.Game, *players: Player
) -> Dict[Player, Dict[Tuple[Infostate, Action], SequenceAttr]]:
    if not players:
        players = list(range(game.num_players()))
    spaces: Dict[Player, Dict[Tuple[Infostate, Action], SequenceAttr]] = {}

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
                            DepthInformedSequence(infostate, action, depth + 1)
                        )
                        stack[actual_stack_length + i] = (
                            state.child(action),
                            plan_copy,
                            depth + 1,
                        )
        spaces[player] = {
            seq: SequenceAttr(root_to_infostate_action_list, sequence_succ_set[seq])
            for seq, root_to_infostate_action_list in sequences.items()
        }
    return spaces


def reduced_normal_form_strategy_space(
    game: pyspiel.Game, *players: Player
) -> Dict[Player, Set[NormalFormPlan]]:
    if not players:
        players = list(range(game.num_players()))
    spaces = {}
    sequences_per_player = sequence_space(game, *players)
    for player, sequences in sequences_per_player.items():
        # the action list are all of that player's actions leading up the infostate I and including the action to
        # take at I.
        # root sequences are those for which the action sequence is only the action to take at I, hence length 1
        root_infostates = set(
            infostate
            for (infostate, _), _ in filter(
                lambda x: len(x[1].action_seq) == 1, sequences.items()
            )
        )
        # we reverse engineer the legal action set of each infostate from all sequences in the sequence space.
        # This could, arguably, be done more efficiently by storing the legal action set during the sequence generation
        available_actions = defaultdict(list)
        for (infostate, action), action_list in sequences.items():
            available_actions[infostate].append(action)

        # infostate_sequences will hold all the possible sequences of
        #   (infostate, action) -> (successor_infostate, successor_action)
        # from each root infostate to the last infostate at which to act for this player.
        # The dictionary will list all combinations of each action at an infostate and each
        # possible immediate successor infostate following this action.
        infostate_sequences = {s: [] for s in root_infostates}
        for root_infostate, seq_list in infostate_sequences.items():
            stack = [(root_infostate, [], 0)]
            while stack:
                infostate, seq, depth = stack.pop()
                for action in available_actions[infostate]:
                    seq_copy = seq.copy()
                    seq_copy.append(DepthInformedSequence(infostate, action, depth))

                    if succ_istates := sequences[(infostate, action)].succ_infostates:
                        stack.extend(
                            [
                                (succ_infostate, seq_copy, depth + 1)
                                for succ_infostate in succ_istates
                            ]
                        )
                    else:
                        seq_list.append(seq_copy)

        spaces[player] = set(
            tuple(
                (infostate, action)
                for infostate, action, _ in sorted(
                    itertools.chain(*seq_lists_combo), key=depth_informed_seq_filter
                )
            )
            for seq_lists_combo in itertools.product(*infostate_sequences.values())
        )
    return spaces


def normal_form_expected_payoff(
    game: pyspiel.Game,
    joint_plan: Union[List[NormalFormPlan], Dict[Infostate, Action], NormalFormPlan],
):
    if isinstance(joint_plan, list):
        joint_plan = {
            infostate: action for infostate, action in itertools.chain(*joint_plan)
        }
    elif isinstance(joint_plan, tuple):
        joint_plan = {infostate: action for infostate, action in joint_plan}
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
            if infostate not in joint_plan:
                # the infostate not being part of the plan means this is a reduced plan and the current point is
                # unreachable by the player due to previous decisions. The expected outcome of this is hence simply 0,
                # since it is never reached.
                continue
            s.apply_action(joint_plan[infostate])
            stack.append((s, chance))
    return expected_payoff


def normal_form_expected_payoff_table(
    game: pyspiel.Game, strategy_spaces: Sequence[NormalFormStrategySpace]
):
    payoffs: Dict[Tuple[NormalFormPlan], Sequence[float]] = dict()
    for joint_profile in itertools.product(*strategy_spaces):
        payoffs[joint_profile] = normal_form_expected_payoff(
            game,
            {
                infostate: action
                for infostate, action in itertools.chain(*joint_profile)
            },
        )
    return payoffs


def reachable_terminal_states(
    game: pyspiel.Game,
    players: Optional[Sequence[Player]] = None,
    plans: Optional[Mapping[Player, Set[NormalFormPlan]]] = None,
    use_progressbar: bool = False,
):
    if players is None:
        players = list(range(game.num_players()))
    if plans is None:
        plans = reduced_normal_form_strategy_space(game, *players)

    reachable_labels_map: Dict[NormalFormPlan, List[str]] = dict()
    for player in players:
        for plan in (
            tqdm(
                plans[player],
                desc=f"Finding reachable terminal states for player {player}",
            )
            if use_progressbar
            else plans[player]
        ):
            reachable_labels_map.setdefault(plan, [])
            # convert to dictionary for faster lookups
            plan_dict = {infostate: action for infostate, action in plan}

            stack = [game.new_initial_state()]
            while stack:
                s = stack.pop()
                if s.is_chance_node():
                    stack.extend([s.child(action) for action, _ in s.chance_outcomes()])
                elif s.is_terminal():
                    reachable_labels_map[plan].append(str(s))
                else:
                    if (curr_player := s.current_player()) == player:
                        if (
                            action := plan_dict.get(
                                s.information_state_string(curr_player),
                                None,  # substitute if missing
                            )
                        ) is not None:
                            s.apply_action(action)
                            stack.append(s)
                    else:
                        stack.extend([s.child(action) for action in s.legal_actions()])
    return reachable_labels_map


def behaviour_to_normal_form(
    game: pyspiel.Game,
    players: Sequence[Player],
    behavior_strategies: Mapping[
        Player,
        Mapping[
            Infostate, Union[Mapping[Action, Probability], Tuple[Action, Probability]]
        ],
    ],
    *,
    reachable_labels_map: Optional[Mapping[NormalFormPlan, List[str]]] = None,
    plans: Optional[Mapping[Player, Set[NormalFormPlan]]] = None,
):
    if reachable_labels_map is None:
        reachable_labels_map = reachable_terminal_states(game, players, plans)
    if plans is None:
        plans = reduced_normal_form_strategy_space(game, *players)

    terminal_reach_prob = terminal_reach_probabilities(
        game, players, behavior_strategies
    )

    plans_out = dict()
    for player in players:
        plan_out = []
        while terminal_reach_prob:
            argmax_plan, max_value = tuple(), -float("inf")
            for plan in plans[player]:
                minimal_prob = min(
                    [terminal_reach_prob[z] for z in reachable_labels_map[plan]]
                )
                if minimal_prob > max_value:
                    argmax_plan = plan
                    max_value = minimal_prob
            plan_out.append((argmax_plan, max_value))
            for label in terminal_reach_prob[argmax_plan]:
                terminal_reach_prob[label] -= max_value
    return plans_out


def terminal_reach_probabilities(
    game: pyspiel.Game,
    players: Sequence[Player],
    behavior_strategies: Mapping[
        Player, Mapping[Infostate, Mapping[Action, Probability]]
    ],
):
    root = game.new_initial_state()
    stack = [(root, {p: 1.0 for p in players})]
    terminal_reach_prob = dict()
    while stack:
        state, reach_prob = stack.pop()
        if state.is_chance_node():
            for outcome, _ in state.chance_outcomes():
                stack.append((state.child(outcome), reach_prob))
        elif state.is_terminal():
            terminal_reach_prob[state] = reach_prob
        else:
            if (curr_player := state.current_player()) in players:
                infostate = state.information_state_string(curr_player)
                policy = behavior_strategies[curr_player][infostate]
                for action in state.legal_actions():
                    child_reach_prob = deepcopy(reach_prob)
                    child_reach_prob[curr_player] *= policy[action]
                    stack.append((state.child(action), child_reach_prob))
            else:
                for action in state.legal_actions():
                    stack.append((state.child(action), reach_prob))
    return terminal_reach_prob
