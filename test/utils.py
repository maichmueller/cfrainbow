from typing import Any, Iterable, Iterator, List, Optional


class CircularList:
    def __init__(self, size: int, default: Optional[Any] = None) -> None:
        self._size: int = size
        self._default: Optional[Any] = default
        self._items: List[Any] = [default] * size
        self._cursor: int = 0

    def push(self, *items: Any) -> None:
        for item in items:
            self._items[self._cursor] = item
            self._cursor = (self._cursor + 1) % self._size

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator:
        return iter(self._items)

    def __getitem__(self, index: int) -> Any:
        if index >= len(self._items):
            raise IndexError("list index out of range")
        actual_index = (self._cursor - 1 - index) % self._size
        return self._items[actual_index]

    def __repr__(self):
        return repr(self._items)
