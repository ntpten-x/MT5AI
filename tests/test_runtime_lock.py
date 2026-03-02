from __future__ import annotations

from modules.runtime_lock import SingleInstanceLock


def test_single_instance_lock_rejects_second_holder(tmp_path):
    lock_path = tmp_path / "run-live.lock"
    first = SingleInstanceLock(lock_path)
    second = SingleInstanceLock(lock_path)

    assert first.acquire() is True
    try:
        assert second.acquire() is False
    finally:
        first.release()
