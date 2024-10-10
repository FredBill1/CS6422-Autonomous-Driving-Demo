from typing import Protocol


class SupportsBool(Protocol):
    def __bool__(self) -> bool: ...
