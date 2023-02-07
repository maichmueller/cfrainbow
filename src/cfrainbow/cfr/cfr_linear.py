from .cfr_discounted import DiscountedCFR


class LinearCFR(DiscountedCFR):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            dict(
                alpha=1.0,
                beta=1.0,
                gamma=1.0,
            )
        )
        super().__init__(*args, **kwargs)
