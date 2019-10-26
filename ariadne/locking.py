import tempfile
from pathlib import Path
from typing import ContextManager

from filelock import FileLock


class LockManager:
    def __init__(self):
        self._lock_directory: Path = Path(tempfile.gettempdir())

    def lock_model(self, classifier_name: str, user_id: str) -> ContextManager:
        return self._get_lock(classifier_name, user_id)

    def is_training(self, classifier_name: str, user_id: str) -> bool:
        lock = self._get_lock(classifier_name, user_id)
        return lock.is_locked

    def _get_lock(self, classifier_name: str, user_id: str) -> FileLock:
        lock_path = self._lock_directory / f"{classifier_name}_{user_id}.lock"
        return FileLock(lock_path, timeout=1)
