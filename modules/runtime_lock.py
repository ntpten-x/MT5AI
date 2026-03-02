from __future__ import annotations

import os
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows path
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX path
    msvcrt = None


class SingleInstanceLock:
    def __init__(self, path: Path):
        self.path = path
        self._handle = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "a+", encoding="ascii")
        self._handle.seek(0)

        try:
            if msvcrt is not None:  # pragma: no branch - platform-specific
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            elif fcntl is not None:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:  # pragma: no cover - unsupported platform
                raise RuntimeError("No file-locking implementation available")
        except OSError:
            self._handle.close()
            self._handle = None
            return False

        self._handle.truncate(0)
        self._handle.write(str(os.getpid()))
        self._handle.flush()
        return True

    def release(self) -> None:
        if self._handle is None:
            return

        self._handle.seek(0)
        try:
            if msvcrt is not None:  # pragma: no branch - platform-specific
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            elif fcntl is not None:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "SingleInstanceLock":
        if not self.acquire():
            raise RuntimeError(f"Another process already holds lock {self.path}")
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.release()
