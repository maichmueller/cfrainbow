from .cfr_base import CFRBase


class CFRLazy(CFRBase):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        kwargs.update(
            dict(
                # only allow alternating update cycles
                alternating=True,
            )
        )
        super().__init__(
            *args,
            **kwargs,
        )
        raise NotImplementedError("Lazy CFR is not yet implemented/")
