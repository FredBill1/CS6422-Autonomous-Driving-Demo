from multiprocessing.connection import Connection
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal


class PipeRecvWorker(QThread):
    recv = Signal(object)

    def __init__(self, pipe: Connection, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._pipe = pipe

    def run(self) -> None:
        while True:
            self.recv.emit(self._pipe.recv())
