import itertools
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
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
    Generator,
)
import numpy as np
import pyspiel
from tqdm import tqdm

from cfrainbow.spiel_types import (
    Infostate,
    Action,
    NormalFormPlan,
    NormalFormStrategySpace,
    Probability,
    Player,
    JointNormalFormPlan,
    JointNormalFormStrategy,
    SequenceFormStrategySpace,
)
from cfrainbow.utils import (
    all_states_gen,
    normalize_state_policy,
    SingletonMeta,
    KeyDependantDefaultDict,
)


class DepthInformedSequence(NamedTuple):
    infostate: Infostate
    action: Action
    depth: int


@dataclass
class SequenceAttr:
    action_seq: Tuple[Action]
    succ_infostates: Set[Infostate]


def depth_informed_seq_order(x: DepthInformedSequence):
    # sort by depth first, then by the infostate str comparison
    # return x.depth, x.infostate
    # sort by length of infostate first, then by the alphabetical order of the infostate str
    return len(x.infostate), x.infostate


def infostate_order(istate: Infostate):
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


def nf_space(game: pyspiel.Game, *players: Player) -> Dict[Player, Set[NormalFormPlan]]:
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
                for sorted_plan in sorted(plan, key=depth_informed_seq_order)
            )
            for plan in itertools.product(
                *sorted(
                    action_space,
                    key=depth_informed_seq_order,
                )
            )
        )
    return strategy_spaces


def sequence_space_gen(
    game: pyspiel.Game, player: Player
) -> Generator[
    Tuple[Tuple[Infostate, Action], Tuple[List[Action], Optional[Infostate]]],
    None,
    None,
]:
    root = game.new_initial_state()
    stack = [(root.clone(), [], 0)]
    while stack:
        state, plan, depth = stack.pop()
        if state.is_terminal():
            n_seqs = len(plan)
            action_seq: List[Action] = []
            for i, seq in enumerate(plan):
                action_seq.append(seq.action)
                yield (
                    (seq.infostate, seq.action),
                    (
                        tuple(action_seq),
                        plan[i + 1].infostate if i < n_seqs - 1 else None,
                    ),
                )

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


def sequence_space(
    game: pyspiel.Game, *players: Player
) -> Dict[Player, Dict[Tuple[Infostate, Action], SequenceAttr]]:
    if not players:
        players = list(range(game.num_players()))
    spaces: Dict[Player, Dict[Tuple[Infostate, Action], SequenceAttr]] = {}

    for player in players:
        space = {}
        for seq, (action_list, succ_infostate) in sequence_space_gen(game, player):
            if seq not in space:
                space[seq] = SequenceAttr(
                    tuple(action_list),
                    {succ_infostate} if succ_infostate is not None else {},
                )
            else:
                if succ_infostate is not None:
                    space[seq].succ_infostates.add(succ_infostate)
        spaces[player] = space
    return spaces


def reduced_nf_space_gen(
    game: pyspiel.Game,
    *players: Player,
    sequence_spaces: Optional[
        Mapping[Player, Mapping[Tuple[Infostate, Action], SequenceAttr]]
    ] = None,
) -> Generator[NormalFormPlan, None, None]:
    if not players:
        if sequence_spaces is not None:
            players = list(sequence_spaces.keys())
        else:
            players = list(range(game.num_players()))
    if sequence_spaces is None:
        sequence_spaces = sequence_space(game, *players)

    assert all(
        p in sequence_spaces for p in sorted(players)
    ), f"Sequence spaces map does not hold an entry for each player passed."

    for player in players:
        for seq_lists_combo in itertools.product(
            *infostate_sequences(sequence_spaces[player]).values()
        ):
            yield tuple(
                (infostate, action)
                for infostate, action, _ in sorted(
                    itertools.chain(*seq_lists_combo), key=depth_informed_seq_order
                )
            )


def reduced_nf_space(
    game: pyspiel.Game,
    *players: Player,
    sequence_spaces: Optional[
        Mapping[Player, Mapping[Tuple[Infostate, Action], SequenceAttr]]
    ] = None,
) -> Dict[Player, Set[NormalFormPlan]]:
    spaces = {}
    if sequence_spaces is None:
        sequence_spaces = sequence_space(game, *players)
    for player, sequences in sequence_spaces.items():
        spaces[player] = set(
            reduced_nf_space_gen(game, player, sequence_spaces=sequence_spaces)
        )
    return spaces


def infostate_sequences(sequences: Dict[Tuple[Infostate, Action], SequenceAttr]):
    """
    Produces all possible player sequences from root-infostates to their last infostate reachable in that sequence.

    Parameters
    ----------
    sequences: Dict[Tuple[Infostate, Action], SequenceAttr]
        all the sequences available to a player with an attribute tuple containing the associated action history and
        set of  reachable infostates

    Returns
    -------
    Dict[Infostate, List[DepthInformedSequence]]
        the dictionary of root infostates to possible continuation sequences
    """
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
    istate_sequences = {s: [] for s in root_infostates}
    for root_infostate, seq_list in istate_sequences.items():
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
    return istate_sequences


def nf_plan_expected_payoff(
    game: pyspiel.Game,
    joint_plan: Union[JointNormalFormPlan, Dict[Infostate, Action], NormalFormPlan],
):
    if isinstance(joint_plan, JointNormalFormPlan):
        joint_plan = {
            infostate: action
            for infostate, action in itertools.chain(*joint_plan.plans)
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


def nf_strategy_expected_payoff(
    game: pyspiel.Game,
    joint_strategy: JointNormalFormStrategy,
    payoff_table: Optional[Dict[JointNormalFormPlan, Sequence[float]]] = None,
):
    if payoff_table is None:
        payoff_table = KeyDependantDefaultDict(
            lambda joint_plan: nf_expected_payoff_table(game, joint_plan)
        )
    root = game.new_initial_state()
    expected_payoff = np.zeros(root.num_players())
    for joint_strategy in itertools.product(*joint_strategy.strategies):
        j_prob = 1.0
        j_plan = []
        for plan, prob in joint_strategy:
            j_prob *= prob
            j_plan.append(plan)
        expected_payoff += j_prob * np.asarray(
            payoff_table[JointNormalFormPlan(j_plan)]
        )

    return expected_payoff


def nf_expected_payoff_table(
    game: pyspiel.Game, strategy_spaces: Sequence[NormalFormStrategySpace]
):
    payoffs: Dict[Tuple[NormalFormPlan], Sequence[float]] = dict()
    for joint_profile in itertools.product(*strategy_spaces):
        payoffs[joint_profile] = nf_plan_expected_payoff(
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
        plans = reduced_nf_space(game, *players)

    if use_progressbar:
        tqdm_opt = tqdm
    else:
        tqdm_opt = lambda *args, **kwargs: args[0]

    reachable_labels_map: Dict[NormalFormPlan, List[str]] = dict()
    for player in players:
        for plan in tqdm_opt(
            plans[player],
            desc=f"Finding reachable terminal states for player {player}",
        ):
            reachable_labels_map.setdefault(plan, [])
            # convert to dictionary for simpler lookups
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


def behaviour_to_nf_strategy(
    game: pyspiel.Game,
    players: Sequence[Player],
    behaviour_strategies: Mapping[
        Player,
        Mapping[
            Infostate, Union[Mapping[Action, Probability], Tuple[Action, Probability]]
        ],
    ],
    reachable_labels_map: Optional[Mapping[NormalFormPlan, List[str]]] = None,
    plans: Optional[Mapping[Player, Set[NormalFormPlan]]] = None,
):
    if reachable_labels_map is None:
        reachable_labels_map = reachable_terminal_states(game, players, plans)
    if plans is None:
        plans = reduced_nf_space(game, *players)

    terminal_reach_prob = terminal_reach_probabilities(
        game, players, behaviour_strategies
    )

    nf_strategies_out = dict()
    for player in players:
        nf_player_strategy = []
        player_terminal_rp = defaultdict(float)
        player_terminal_rp.update(
            {z: values[player] for z, values in terminal_reach_prob.items()}
        )
        player_plans = plans[player]
        while any(prob > 1e-10 for prob in player_terminal_rp.values()):
            argmax_plan, max_value = tuple(), -float("inf")
            for plan in player_plans:
                minimal_prob = min(
                    player_terminal_rp[z] for z in reachable_labels_map[plan]
                )
                if minimal_prob > max_value:
                    argmax_plan = plan
                    max_value = minimal_prob
            nf_player_strategy.append((argmax_plan, max_value))
            for label in reachable_labels_map[argmax_plan]:
                player_terminal_rp[label] -= max_value
        nf_strategies_out[player] = tuple(nf_player_strategy)
    return nf_strategies_out


def nf_to_behaviour_strategy(
    players: Sequence[Player],
    normal_form_strategies: Mapping[
        Player, Sequence[Tuple[NormalFormPlan, Probability]]
    ],
) -> Dict[Player, Dict[Infostate, Dict[Action, Probability]]]:
    behaviour_strategies_out = dict()
    for player in players:
        behavioural_strategy = dict()
        nf_strategy = tuple(normal_form_strategies[player])
        for plan, probability in nf_strategy:
            for infostate, action in plan:
                behavioural_strategy.setdefault(infostate, dict())
                policy = behavioural_strategy[infostate]
                policy.setdefault(action, 0.0)
                policy[action] += probability
        behaviour_strategies_out[player] = normalize_state_policy(behavioural_strategy)
    return behaviour_strategies_out


def terminal_reach_probabilities(
    game: pyspiel.Game,
    players: Sequence[Player],
    behaviour_strategies: Mapping[
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
            if any(rp > 0.0 for rp in reach_prob.values()):
                terminal_reach_prob[str(state)] = reach_prob
        else:
            if (curr_player := state.current_player()) in players:
                infostate = state.information_state_string(curr_player)
                policy = behaviour_strategies[curr_player][infostate]
                for action in state.legal_actions():
                    child_reach_prob = deepcopy(reach_prob)
                    child_reach_prob[curr_player] *= policy[action]
                    stack.append((state.child(action), child_reach_prob))
            else:
                for action in state.legal_actions():
                    stack.append((state.child(action), reach_prob))
    return terminal_reach_prob


@dataclass
class DecisionSpace:
    normal_form: Optional[Dict[Player, NormalFormStrategySpace]] = None
    reduced_normal_form: Optional[Dict[Player, NormalFormStrategySpace]] = None
    sequence_form: Optional[Dict[Player, SequenceFormStrategySpace]] = None


class SpaceForm(Enum):
    normal = 1
    reduced_normal = 2
    sequence = 3


class MasterSpace(metaclass=SingletonMeta):
    game_space: Dict[str, DecisionSpace] = defaultdict(lambda: DecisionSpace())

    def __class_getitem__(
        cls, game_token: Tuple[Union[pyspiel.Game, str], SpaceForm]
    ) -> DecisionSpace:
        game, token = game_token
        space = cls.game_space[str(game)]
        if getattr(space, token.name, None) is None:
            if token == SpaceForm.normal:
                new_space = nf_space(game, *game.players())
            elif token == SpaceForm.reduced_normal:
                new_space = reduced_nf_space(game, *game.players())
            elif token == SpaceForm.sequence:
                new_space = sequence_space(game, *game.players())
            else:
                raise ValueError(f"Unknown space token {token}.")
            setattr(space, token.name, new_space)
        return getattr(space, token.name)

    @classmethod
    def __contains__(cls, game: Union[pyspiel.Game, str]) -> bool:
        return str(game) in cls.game_space
