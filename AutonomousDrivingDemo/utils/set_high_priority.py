from typing import Optional

import psutil


def set_high_priority(pid: Optional[int] = None) -> None:
    psutil.Process(pid).nice(psutil.REALTIME_PRIORITY_CLASS if psutil.WINDOWS else -20)
