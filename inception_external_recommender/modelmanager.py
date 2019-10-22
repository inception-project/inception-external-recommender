import os
import tempfile
from pathlib import Path
from typing import Any, Optional, ContextManager

from filelock import Timeout, FileLock

import joblib


class ModelManager:

    def __init__(self, directory: Optional[str] = None):
        if directory is None:
            directory = Path(__file__).resolve().parents[1]
        else:
            directory = Path(directory)

        self._directory: Path = directory

    def lock_model(self, classifier_name: str, user_id: str) -> ContextManager:
        return self._get_lock(classifier_name, user_id)

    def is_training(self, classifier_name: str, user_id: str) -> bool:
        lock = self._get_lock(classifier_name, user_id)
        return lock.is_locked

    def load_model(self, classifier_name: str, user_id: str) -> Optional[Any]:
        model_path = self._get_model_path(classifier_name, user_id)
        return joblib.load(model_path)

    def save_model(self, classifier_name: str, user_id: str, model: Any):
        model_path = self._get_model_path(classifier_name, user_id)
        tmp_model_path = model_path.with_suffix(".joblib.tmp")
        joblib.dump(model, tmp_model_path)
        os.replace(tmp_model_path, model_path)

    def _get_model_path(self, classifier_name: str, user_id: str) -> Path:
        return self._directory / classifier_name / f"model_{user_id}.joblib"

    def _get_lock(self, classifier_name: str, user_id: str) -> FileLock:
        model_path = self._get_model_path(classifier_name, user_id)
        lock_path = model_path.with_suffix(".lock")
        return FileLock(lock_path)
