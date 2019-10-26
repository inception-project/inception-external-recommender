import logging
import os
from pathlib import Path
from typing import Any, Optional

import joblib

logger = logging.getLogger(__file__)


class ModelManager:
    def __init__(self, directory: Optional[str] = None):
        if directory is None:
            directory = Path(__file__).resolve().parents[1] / "models"
        else:
            directory = Path(directory)

        self._directory: Path = directory

    def load_model(self, classifier_name: str, user_id: str) -> Optional[Any]:
        model_path = self._get_model_path(classifier_name, user_id)
        if model_path.is_file():
            logger.debug("Model found for [%s]", model_path)
            return joblib.load(model_path)
        else:
            logger.debug("No model found for [%s]", model_path)
            return None

    def save_model(self, classifier_name: str, user_id: str, model: Any):
        model_path = self._get_model_path(classifier_name, user_id)
        tmp_model_path = model_path.with_suffix(".joblib.tmp")
        joblib.dump(model, tmp_model_path)
        os.replace(tmp_model_path, model_path)

    def _get_model_path(self, classifier_name: str, user_id: str) -> Path:
        return self._directory / classifier_name / f"model_{user_id}.joblib"
