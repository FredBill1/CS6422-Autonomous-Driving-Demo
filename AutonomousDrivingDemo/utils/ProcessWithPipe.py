import multiprocessing as mp
from multiprocessing.connection import Connection
from typing import Any, Optional, Protocol, override

from PySide6.QtCore import QObject, QThread, Signal

from .set_high_priority import set_high_priority


class CallableWithConnection(Protocol):
    def __call__(self, pipe: Connection, *args: Any, **kwargs: Any) -> None: ...


class ProcessWithPipe(QThread):
    recv = Signal(object)

    def __init__(self, target: CallableWithConnection, args=(), kwargs={}, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._parent_pipe, self._child_pipe = mp.Pipe()
        self._process = mp.Process(target=target, args=(self._child_pipe, *args), kwargs=kwargs, daemon=True)

    @override
    def run(self) -> None:
        self._process.start()
        set_high_priority(self._process.pid)
        while True:
            self.recv.emit(self._parent_pipe.recv())

    def send(self, obj: Any) -> None:
        self._parent_pipe.send(obj)
