import itertools
from collections import defaultdict
from typing import Optional, Dict, Mapping, List

from cfrainbow.spiel_types import NormalFormPlan, JointNormalFormPlan
from cfrainbow.strategy import (
    reachable_terminal_states,
    reduced_nf_space,
    behaviour_to_nf_strategy,
)
from cfrainbow.utils import infostates_gen
from .cfr_vanilla import VanillaCFR


class JointReconstructionCFR(VanillaCFR):
    def __init__(
        self,
        *args,
        conversion_frequency: int = 1,
        reachable_labels_map=None,
        reduced_normal_form_strategy_space=None,
        **kwargs,
    ):
        kwargs["alternating"] = False
        super().__init__(*args, **kwargs)
        # the storage of all played strategy profiles in the iterations.
        self.empirical_freq_of_play: Dict[JointNormalFormPlan, float] = defaultdict(
            float
        )
        # after how many rounds of vanilla cfr we convert the current strategy to normal-form again.
        # This counts FULL iteration steps, i.e. for n players we have that every n algorithm iterations are 1
        # conversion relevant iteration, so that each player has seen their current strategies equally often updated.
        self._conversion_frequency = conversion_frequency * self.nr_players
        self._underlying_game = self.root_state.get_game()
        self._reduced_nf_space = (
            reduced_normal_form_strategy_space
            if reduced_normal_form_strategy_space is not None
            else reduced_nf_space(self._underlying_game, *self.players)
        )
        self._reachable_labels: Mapping[NormalFormPlan, List[str]] = (
            reachable_labels_map
            if reachable_labels_map is not None
            else reachable_terminal_states(self._underlying_game, self.players)
        )
        self._infostate_active_player = {
            infostate: player
            for infostate, player, _, _ in infostates_gen(root=self.root_state.clone())
        }

    def iterate(
        self,
        updating_player: Optional[int] = None,
    ):
        # first let vanilla cfr run its course
        super().iterate(updating_player)
        # the parent class will also increment the iteration counter which signals the regret minimizers
        # to update the current strategy on the next demand, i.e. in the following

        # now convert the last iteration's behaviour strategies to normal-form
        if self.iteration % self._conversion_frequency == 0:
            curr_behaviour_policy = {p: dict() for p in self.players}
            for infostate, regret_minimizer in self._regret_minimizer_dict.items():
                owner = self._infostate_active_player[infostate]
                curr_behaviour_policy[owner][infostate] = regret_minimizer.recommend(
                    self.iteration
                )
            normal_form_policy_profile = tuple(
                behaviour_to_nf_strategy(
                    self._underlying_game,
                    self.players,
                    curr_behaviour_policy,
                    reachable_labels_map=self._reachable_labels,
                    plans=self._reduced_nf_space,
                ).values()
            )
            jo_p = 0.0
            for combo in itertools.product(*normal_form_policy_profile):
                strategies, joint_prob = [], 1.0
                for player_strategy, prob in combo:
                    strategies.append(player_strategy)
                    joint_prob *= prob
                jo_p += joint_prob
                print(jo_p)
                self.empirical_freq_of_play[
                    JointNormalFormPlan(tuple(strategies))
                ] += joint_prob
            print("Iteration", self.iteration)
            print(
                sum(
                    [
                        v / (self.iteration / self._conversion_frequency)
                        for v in self.empirical_freq_of_play.values()
                    ]
                )
            )

    def empirical_frequency_of_play(self):
        return tuple(
            (plan, probability / (self.iteration / self._conversion_frequency))
            for plan, probability in self.empirical_freq_of_play.items()
        )
