import pyspiel
from .custom_efg import (
    GameZeroSum,
    GameGeneralSum,
    _GAME_TYPE,
    _GAME_TYPE_GENERAL_SUM,
    State,
)


pyspiel.register_game(_GAME_TYPE, GameZeroSum)
pyspiel.register_game(_GAME_TYPE_GENERAL_SUM, GameGeneralSum)

__all__ = [
    "GameZeroSum",
    "GameGeneralSum",
    "State",
]
