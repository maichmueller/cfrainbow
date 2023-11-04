import logging

import cfrainbow.rm as rm

from .cfr_discounted import DiscountedCFR


class PredictivePlusCFR(DiscountedCFR):
    def __init__(
        self,
        *args,
        alpha=float("inf"),
        beta=-float("inf"),
        gamma=1,
        **kwargs,
    ):
        kwargs.update(
            dict(
                # only allow alternating update cycles
                alternating=True,
                regret_minimizer_type=rm.AutoPredictiveRegretMatcherPlus,
            )
        )
        super().__init__(
            *args,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            **kwargs,
        )
        logging.warning(
            "Predictive CFR+ is not reproducing the respective paper results. Use with caution."
        )
