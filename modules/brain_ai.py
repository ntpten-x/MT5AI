from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from modules.feature_engineering import FEATURE_COLUMNS
from modules.timeframes import pandas_frequency


class XGBoostSignalModel:
    def __init__(
        self,
        artifact_path: Path | None,
        feature_columns: Sequence[str] | None = None,
        target_side: str = "long",
    ):
        self.artifact_path = artifact_path
        self.feature_columns = list(feature_columns or FEATURE_COLUMNS)
        self.target_side = target_side
        self.model = None
        self.calibrator = None
        self.calibration_method: str | None = None
        self.threshold: float | None = None

    def _deps(self):
        try:
            import joblib
            import xgboost as xgb
            from sklearn.isotonic import IsotonicRegression
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import (
                accuracy_score,
                brier_score_loss,
                log_loss,
                precision_score,
                recall_score,
                roc_auc_score,
            )
        except ImportError as exc:
            raise RuntimeError("Install project dependencies before training or inference") from exc
        return (
            joblib,
            xgb,
            IsotonicRegression,
            LogisticRegression,
            accuracy_score,
            brier_score_loss,
            log_loss,
            precision_score,
            recall_score,
            roc_auc_score,
        )

    def is_ready(self) -> bool:
        return self.model is not None or (self.artifact_path is not None and self.artifact_path.exists())

    def fit(self, training_frame: pd.DataFrame) -> None:
        if len(training_frame) < 100:
            raise ValueError("Not enough rows to train XGBoost")

        _, xgb, *_ = self._deps()
        self.calibrator = None
        self.calibration_method = None
        self.threshold = None
        x_train = training_frame[self.feature_columns]
        y_train = training_frame["target"].astype(int)

        model = xgb.XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
        )
        model.fit(x_train, y_train)
        self.model = model

    def fit_calibrator(self, calibration_frame: pd.DataFrame, method: str = "auto") -> dict[str, Any]:
        if calibration_frame.empty:
            return {"calibration_method": None, "calibration_rows": 0}
        if self.model is None:
            raise RuntimeError("Model must be fitted before calibration")

        (
            _,
            _,
            IsotonicRegression,
            LogisticRegression,
            _,
            brier_score_loss,
            log_loss,
            _,
            _,
            _,
        ) = self._deps()

        x_cal = calibration_frame[self.feature_columns]
        y_cal = calibration_frame["target"].astype(int)
        raw_probabilities = self.model.predict_proba(x_cal)[:, 1]
        chosen_method = method.lower()
        positives = int(y_cal.sum())
        negatives = int(len(y_cal) - positives)
        if chosen_method == "auto":
            chosen_method = "isotonic" if positives >= 50 and negatives >= 50 else "sigmoid"

        if y_cal.nunique() < 2:
            self.calibrator = None
            self.calibration_method = "identity"
            return {
                "calibration_method": self.calibration_method,
                "calibration_rows": int(len(calibration_frame)),
                "brier_raw": float(brier_score_loss(y_cal, raw_probabilities)),
                "log_loss_raw": float(log_loss(y_cal, raw_probabilities, labels=[0, 1])),
            }

        if chosen_method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(raw_probabilities, y_cal)
        else:
            calibrator = LogisticRegression()
            calibrator.fit(raw_probabilities.reshape(-1, 1), y_cal)
            chosen_method = "sigmoid"

        self.calibrator = calibrator
        self.calibration_method = chosen_method
        calibrated = self._apply_calibrator(raw_probabilities)
        return {
            "calibration_method": self.calibration_method,
            "calibration_rows": int(len(calibration_frame)),
            "brier_raw": float(brier_score_loss(y_cal, raw_probabilities)),
            "brier_calibrated": float(brier_score_loss(y_cal, calibrated)),
            "log_loss_raw": float(log_loss(y_cal, raw_probabilities, labels=[0, 1])),
            "log_loss_calibrated": float(log_loss(y_cal, calibrated, labels=[0, 1])),
        }

    def _apply_calibrator(self, raw_probabilities: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            return np.clip(raw_probabilities, 1e-6, 1 - 1e-6)
        if self.calibration_method == "isotonic":
            calibrated = self.calibrator.predict(raw_probabilities)
        else:
            calibrated = self.calibrator.predict_proba(raw_probabilities.reshape(-1, 1))[:, 1]
        return np.clip(np.asarray(calibrated, dtype=float), 1e-6, 1 - 1e-6)

    def set_threshold(self, threshold: float | None) -> None:
        self.threshold = float(threshold) if threshold is not None else None

    def effective_threshold(self, default: float) -> float:
        return float(self.threshold) if self.threshold is not None else float(default)

    def save(self) -> None:
        if self.artifact_path is None or self.model is None:
            return
        joblib, *_ = self._deps()
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "calibrator": self.calibrator,
                "calibration_method": self.calibration_method,
                "features": self.feature_columns,
                "threshold": self.threshold,
                "target_side": self.target_side,
            },
            self.artifact_path,
        )

    def evaluate(self, evaluation_frame: pd.DataFrame, threshold: float = 0.5) -> dict[str, Any]:
        (
            _,
            _,
            _,
            _,
            accuracy_score,
            brier_score_loss,
            log_loss,
            precision_score,
            recall_score,
            roc_auc_score,
        ) = self._deps()
        if evaluation_frame.empty:
            return {}
        y_true = evaluation_frame["target"].astype(int)
        probabilities = self.predict_proba_frame(evaluation_frame)
        predictions = (probabilities >= threshold).astype(int)

        metrics: dict[str, Any] = {
            "rows_eval": int(len(evaluation_frame)),
            "positive_rate_eval": float(y_true.mean()) if len(y_true) else 0.0,
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(y_true, predictions)),
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "brier_score": float(brier_score_loss(y_true, probabilities)),
            "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        }
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities)) if y_true.nunique() > 1 else None
        return metrics

    def train(
        self,
        training_frame: pd.DataFrame,
        calibration_frame: pd.DataFrame | None = None,
        evaluation_frame: pd.DataFrame | None = None,
        calibration_method: str = "auto",
        threshold: float | None = None,
    ) -> dict[str, Any]:
        self.fit(training_frame)
        calibration_summary = (
            self.fit_calibrator(calibration_frame, method=calibration_method)
            if calibration_frame is not None
            else {"calibration_method": None, "calibration_rows": 0}
        )
        if threshold is not None:
            self.set_threshold(threshold)

        metrics: dict[str, Any] = {
            "rows_train": int(len(training_frame)),
            "positive_rate_train": float(training_frame["target"].mean()),
            **calibration_summary,
        }
        if evaluation_frame is not None:
            metrics.update(self.evaluate(evaluation_frame, threshold=self.effective_threshold(0.5)))

        self.save()
        if self.artifact_path is not None:
            metrics["artifact_path"] = str(self.artifact_path)

        return metrics

    def load(self) -> bool:
        if self.model is not None:
            return True
        if self.artifact_path is None or not self.artifact_path.exists():
            return False
        joblib, *_ = self._deps()
        payload = joblib.load(self.artifact_path)
        self.model = payload["model"]
        self.calibrator = payload.get("calibrator")
        self.calibration_method = payload.get("calibration_method")
        self.feature_columns = payload.get("features", self.feature_columns)
        self.threshold = payload.get("threshold")
        self.target_side = payload.get("target_side", self.target_side)
        return True

    def predict_raw_proba_frame(self, feature_frame: pd.DataFrame) -> pd.Series:
        if not self.load():
            raise RuntimeError("XGBoost model artifact is not available")
        probabilities = self.model.predict_proba(feature_frame[self.feature_columns])[:, 1]
        return pd.Series(np.clip(probabilities, 1e-6, 1 - 1e-6), index=feature_frame.index, name="xgb_raw_probability")

    def predict_proba_frame(self, feature_frame: pd.DataFrame) -> pd.Series:
        if not self.load():
            raise RuntimeError("XGBoost model artifact is not available")
        raw_probabilities = self.model.predict_proba(feature_frame[self.feature_columns])[:, 1]
        probabilities = self._apply_calibrator(raw_probabilities)
        return pd.Series(probabilities, index=feature_frame.index, name="xgb_probability")

    def predict_latest(self, feature_frame: pd.DataFrame) -> float:
        probabilities = self.predict_proba_frame(feature_frame)
        return float(probabilities.iloc[-1])


class LightGBMSignalModel:
    def __init__(
        self,
        artifact_path: Path | None,
        feature_columns: Sequence[str] | None = None,
        target_side: str = "long",
    ):
        self.artifact_path = artifact_path
        self.feature_columns = list(feature_columns or FEATURE_COLUMNS)
        self.target_side = target_side
        self.model = None
        self.calibrator = None
        self.calibration_method: str | None = None
        self.threshold: float | None = None

    def _deps(self):
        try:
            import joblib
            import lightgbm as lgb
            from sklearn.isotonic import IsotonicRegression
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import (
                accuracy_score,
                brier_score_loss,
                log_loss,
                precision_score,
                recall_score,
                roc_auc_score,
            )
        except ImportError as exc:
            raise RuntimeError("Install LightGBM dependencies before training or inference") from exc
        return (
            joblib,
            lgb,
            IsotonicRegression,
            LogisticRegression,
            accuracy_score,
            brier_score_loss,
            log_loss,
            precision_score,
            recall_score,
            roc_auc_score,
        )

    def is_ready(self) -> bool:
        return self.model is not None or (self.artifact_path is not None and self.artifact_path.exists())

    def fit(self, training_frame: pd.DataFrame) -> None:
        if len(training_frame) < 100:
            raise ValueError("Not enough rows to train LightGBM")

        _, lgb, *_ = self._deps()
        self.calibrator = None
        self.calibration_method = None
        self.threshold = None
        x_train = training_frame[self.feature_columns]
        y_train = training_frame["target"].astype(int)
        model = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(x_train, y_train)
        self.model = model

    def fit_calibrator(self, calibration_frame: pd.DataFrame, method: str = "auto") -> dict[str, Any]:
        if calibration_frame.empty:
            return {"calibration_method": None, "calibration_rows": 0}
        if self.model is None:
            raise RuntimeError("Model must be fitted before calibration")

        (
            _,
            _,
            IsotonicRegression,
            LogisticRegression,
            _,
            brier_score_loss,
            log_loss,
            _,
            _,
            _,
        ) = self._deps()

        x_cal = calibration_frame[self.feature_columns]
        y_cal = calibration_frame["target"].astype(int)
        raw_probabilities = self.model.predict_proba(x_cal)[:, 1]
        chosen_method = method.lower()
        positives = int(y_cal.sum())
        negatives = int(len(y_cal) - positives)
        if chosen_method == "auto":
            chosen_method = "isotonic" if positives >= 50 and negatives >= 50 else "sigmoid"

        if y_cal.nunique() < 2:
            self.calibrator = None
            self.calibration_method = "identity"
            return {
                "calibration_method": self.calibration_method,
                "calibration_rows": int(len(calibration_frame)),
                "brier_raw": float(brier_score_loss(y_cal, raw_probabilities)),
                "log_loss_raw": float(log_loss(y_cal, raw_probabilities, labels=[0, 1])),
            }

        if chosen_method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(raw_probabilities, y_cal)
        else:
            calibrator = LogisticRegression()
            calibrator.fit(raw_probabilities.reshape(-1, 1), y_cal)
            chosen_method = "sigmoid"

        self.calibrator = calibrator
        self.calibration_method = chosen_method
        calibrated = self._apply_calibrator(raw_probabilities)
        return {
            "calibration_method": self.calibration_method,
            "calibration_rows": int(len(calibration_frame)),
            "brier_raw": float(brier_score_loss(y_cal, raw_probabilities)),
            "brier_calibrated": float(brier_score_loss(y_cal, calibrated)),
            "log_loss_raw": float(log_loss(y_cal, raw_probabilities, labels=[0, 1])),
            "log_loss_calibrated": float(log_loss(y_cal, calibrated, labels=[0, 1])),
        }

    def _apply_calibrator(self, raw_probabilities: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            return np.clip(raw_probabilities, 1e-6, 1 - 1e-6)
        if self.calibration_method == "isotonic":
            calibrated = self.calibrator.predict(raw_probabilities)
        else:
            calibrated = self.calibrator.predict_proba(raw_probabilities.reshape(-1, 1))[:, 1]
        return np.clip(np.asarray(calibrated, dtype=float), 1e-6, 1 - 1e-6)

    def set_threshold(self, threshold: float | None) -> None:
        self.threshold = float(threshold) if threshold is not None else None

    def effective_threshold(self, default: float) -> float:
        return float(self.threshold) if self.threshold is not None else float(default)

    def save(self) -> None:
        if self.artifact_path is None or self.model is None:
            return
        joblib, *_ = self._deps()
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "calibrator": self.calibrator,
                "calibration_method": self.calibration_method,
                "features": self.feature_columns,
                "threshold": self.threshold,
                "target_side": self.target_side,
            },
            self.artifact_path,
        )

    def evaluate(self, evaluation_frame: pd.DataFrame, threshold: float = 0.5) -> dict[str, Any]:
        (
            _,
            _,
            _,
            _,
            accuracy_score,
            brier_score_loss,
            log_loss,
            precision_score,
            recall_score,
            roc_auc_score,
        ) = self._deps()
        if evaluation_frame.empty:
            return {}
        y_true = evaluation_frame["target"].astype(int)
        probabilities = self.predict_proba_frame(evaluation_frame)
        predictions = (probabilities >= threshold).astype(int)
        metrics: dict[str, Any] = {
            "rows_eval": int(len(evaluation_frame)),
            "positive_rate_eval": float(y_true.mean()) if len(y_true) else 0.0,
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(y_true, predictions)),
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "brier_score": float(brier_score_loss(y_true, probabilities)),
            "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        }
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities)) if y_true.nunique() > 1 else None
        return metrics

    def train(
        self,
        training_frame: pd.DataFrame,
        calibration_frame: pd.DataFrame | None = None,
        evaluation_frame: pd.DataFrame | None = None,
        calibration_method: str = "auto",
        threshold: float | None = None,
    ) -> dict[str, Any]:
        self.fit(training_frame)
        calibration_summary = (
            self.fit_calibrator(calibration_frame, method=calibration_method)
            if calibration_frame is not None
            else {"calibration_method": None, "calibration_rows": 0}
        )
        if threshold is not None:
            self.set_threshold(threshold)
        metrics: dict[str, Any] = {
            "rows_train": int(len(training_frame)),
            "positive_rate_train": float(training_frame["target"].mean()),
            **calibration_summary,
        }
        if evaluation_frame is not None:
            metrics.update(self.evaluate(evaluation_frame, threshold=self.effective_threshold(0.5)))
        self.save()
        if self.artifact_path is not None:
            metrics["artifact_path"] = str(self.artifact_path)
        return metrics

    def load(self) -> bool:
        if self.model is not None:
            return True
        if self.artifact_path is None or not self.artifact_path.exists():
            return False
        joblib, *_ = self._deps()
        payload = joblib.load(self.artifact_path)
        self.model = payload["model"]
        self.calibrator = payload.get("calibrator")
        self.calibration_method = payload.get("calibration_method")
        self.feature_columns = payload.get("features", self.feature_columns)
        self.threshold = payload.get("threshold")
        self.target_side = payload.get("target_side", self.target_side)
        return True

    def predict_proba_frame(self, feature_frame: pd.DataFrame) -> pd.Series:
        if not self.load():
            raise RuntimeError("LightGBM model artifact is not available")
        raw_probabilities = self.model.predict_proba(feature_frame[self.feature_columns])[:, 1]
        probabilities = self._apply_calibrator(raw_probabilities)
        return pd.Series(probabilities, index=feature_frame.index, name="lgbm_probability")

    def predict_latest(self, feature_frame: pd.DataFrame) -> float:
        probabilities = self.predict_proba_frame(feature_frame)
        return float(probabilities.iloc[-1])


class TorchSequenceSignalModel:
    def __init__(
        self,
        artifact_path: Path | None,
        feature_columns: Sequence[str] | None = None,
        target_side: str = "long",
        sequence_length: int = 48,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        epochs: int = 8,
        batch_size: int = 64,
        device: str = "cpu",
        architecture: str = "lstm",
        transformer_heads: int = 4,
        transformer_ffn_size: int = 128,
    ):
        self.artifact_path = artifact_path
        self.feature_columns = list(feature_columns or FEATURE_COLUMNS)
        self.target_side = target_side
        self.sequence_length = int(sequence_length)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.device = device
        self.architecture = str(architecture).strip().lower()
        self.transformer_heads = int(transformer_heads)
        self.transformer_ffn_size = int(transformer_ffn_size)
        self.model = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None

    def _deps(self):
        try:
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise RuntimeError("Install torch before training or inference with the sequence model") from exc
        return torch, nn, DataLoader, TensorDataset

    def is_ready(self) -> bool:
        return self.model is not None or (self.artifact_path is not None and self.artifact_path.exists())

    def _build_model(self, input_size: int):
        torch, nn, *_ = self._deps()

        class _LSTMClassifier(nn.Module):
            def __init__(self, feature_count: int, hidden_size: int, num_layers: int, dropout: float):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=feature_count,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True,
                )
                projection = max(16, hidden_size // 2)
                self.head = nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, projection),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(projection, 1),
                )

            def forward(self, features):
                sequence, _ = self.lstm(features)
                logits = self.head(sequence[:, -1, :])
                return logits.squeeze(-1)

        class _TransformerClassifier(nn.Module):
            def __init__(
                self,
                feature_count: int,
                hidden_size: int,
                num_layers: int,
                dropout: float,
                num_heads: int,
                ffn_size: int,
                sequence_length: int,
            ):
                super().__init__()
                adjusted_heads = max(1, min(num_heads, hidden_size))
                while hidden_size % adjusted_heads != 0 and adjusted_heads > 1:
                    adjusted_heads -= 1
                self.input_projection = nn.Linear(feature_count, hidden_size)
                self.position_embedding = nn.Parameter(torch.zeros(1, sequence_length, hidden_size))
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=adjusted_heads,
                    dim_feedforward=max(hidden_size * 2, ffn_size),
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, num_layers))
                projection = max(16, hidden_size // 2)
                self.head = nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, projection),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(projection, 1),
                )

            def forward(self, features):
                projected = self.input_projection(features)
                positional = self.position_embedding[:, : projected.size(1), :]
                encoded = self.encoder(projected + positional)
                logits = self.head(encoded[:, -1, :])
                return logits.squeeze(-1)

        if self.architecture == "transformer":
            model = _TransformerClassifier(
                feature_count=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                num_heads=self.transformer_heads,
                ffn_size=self.transformer_ffn_size,
                sequence_length=self.sequence_length,
            )
        else:
            model = _LSTMClassifier(
                feature_count=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
            )
        return model.to(torch.device(self.device))

    def _prepare_arrays(
        self,
        feature_frame: pd.DataFrame,
        fit_statistics: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        features = feature_frame[self.feature_columns].to_numpy(dtype=np.float32)
        if fit_statistics or self.feature_mean is None or self.feature_std is None:
            self.feature_mean = features.mean(axis=0).astype(np.float32)
            self.feature_std = features.std(axis=0).astype(np.float32)
            self.feature_std[self.feature_std < 1e-6] = 1.0
        normalized = (features - self.feature_mean) / self.feature_std
        return features, normalized.astype(np.float32)

    def _windowed_dataset(
        self,
        feature_frame: pd.DataFrame,
        include_target: bool = True,
        fit_statistics: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        if len(feature_frame) < self.sequence_length:
            raise ValueError("Not enough rows to build sequence dataset")
        _, normalized = self._prepare_arrays(feature_frame, fit_statistics=fit_statistics)
        targets = (
            feature_frame["target"].to_numpy(dtype=np.float32)
            if include_target and "target" in feature_frame.columns
            else None
        )
        windows: list[np.ndarray] = []
        labels: list[float] = []
        indices: list[int] = []
        for end in range(self.sequence_length - 1, len(feature_frame)):
            start = end - self.sequence_length + 1
            windows.append(normalized[start : end + 1])
            indices.append(end)
            if targets is not None:
                labels.append(float(targets[end]))
        x_values = np.asarray(windows, dtype=np.float32)
        y_values = np.asarray(labels, dtype=np.float32) if labels else None
        return x_values, y_values, np.asarray(indices, dtype=int)

    def fit(self, training_frame: pd.DataFrame) -> None:
        if len(training_frame) < max(100, self.sequence_length * 3):
            raise ValueError("Not enough rows to train the sequence model")

        torch, nn, DataLoader, TensorDataset = self._deps()
        x_train, y_train, _ = self._windowed_dataset(training_frame, include_target=True, fit_statistics=True)
        dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True)

        model = self._build_model(input_size=x_train.shape[-1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        positives = float(y_train.sum())
        negatives = float(len(y_train) - positives)
        pos_weight = None
        if positives > 0 and negatives > 0:
            pos_weight = torch.tensor([max(1.0, negatives / max(positives, 1.0))], device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        model.train()
        for _ in range(self.epochs):
            for batch_features, batch_targets in loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_features)
                loss = criterion(logits, batch_targets)
                loss.backward()
                optimizer.step()
        self.model = model

    def save(self) -> None:
        if self.artifact_path is None or self.model is None:
            return
        torch, *_ = self._deps()
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "feature_columns": self.feature_columns,
                "target_side": self.target_side,
                "sequence_length": self.sequence_length,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "architecture": self.architecture,
                "transformer_heads": self.transformer_heads,
                "transformer_ffn_size": self.transformer_ffn_size,
                "feature_mean": self.feature_mean,
                "feature_std": self.feature_std,
            },
            self.artifact_path,
        )

    def load(self) -> bool:
        if self.model is not None:
            return True
        if self.artifact_path is None or not self.artifact_path.exists():
            return False
        torch, *_ = self._deps()
        payload = torch.load(self.artifact_path, map_location=self.device, weights_only=False)
        self.feature_columns = list(payload.get("feature_columns", self.feature_columns))
        self.target_side = payload.get("target_side", self.target_side)
        self.sequence_length = int(payload.get("sequence_length", self.sequence_length))
        self.hidden_size = int(payload.get("hidden_size", self.hidden_size))
        self.num_layers = int(payload.get("num_layers", self.num_layers))
        self.dropout = float(payload.get("dropout", self.dropout))
        self.architecture = str(payload.get("architecture", self.architecture)).strip().lower()
        self.transformer_heads = int(payload.get("transformer_heads", self.transformer_heads))
        self.transformer_ffn_size = int(payload.get("transformer_ffn_size", self.transformer_ffn_size))
        self.feature_mean = np.asarray(payload.get("feature_mean"), dtype=np.float32)
        self.feature_std = np.asarray(payload.get("feature_std"), dtype=np.float32)
        self.model = self._build_model(input_size=len(self.feature_columns))
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        return True

    def predict_proba_frame(self, feature_frame: pd.DataFrame) -> pd.Series:
        torch, *_ = self._deps()
        if not self.load():
            raise RuntimeError("Sequence model artifact is not available")
        if len(feature_frame) < self.sequence_length:
            return pd.Series(np.nan, index=feature_frame.index, name="sequence_probability")
        self.model.eval()
        x_values, _, indices = self._windowed_dataset(feature_frame, include_target=False, fit_statistics=False)
        with torch.no_grad():
            logits = self.model(torch.tensor(x_values, device=self.device))
            probabilities = torch.sigmoid(logits).cpu().numpy().astype(float)
        full = pd.Series(np.nan, index=feature_frame.index, name="sequence_probability")
        full.iloc[indices] = probabilities
        return full

    def predict_latest(self, feature_frame: pd.DataFrame) -> float:
        probabilities = self.predict_proba_frame(feature_frame).dropna()
        if probabilities.empty:
            raise RuntimeError("Sequence model did not produce a probability")
        return float(probabilities.iloc[-1])

    def evaluate(self, evaluation_frame: pd.DataFrame, threshold: float = 0.5) -> dict[str, Any]:
        try:
            from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, precision_score, recall_score, roc_auc_score
        except ImportError as exc:
            raise RuntimeError("Install scikit-learn before evaluating the sequence model") from exc
        if len(evaluation_frame) < self.sequence_length:
            return {}
        probabilities = self.predict_proba_frame(evaluation_frame).dropna()
        if probabilities.empty:
            return {}
        truth = evaluation_frame.loc[probabilities.index, "target"].astype(int)
        predictions = (probabilities >= threshold).astype(int)
        metrics: dict[str, Any] = {
            "rows_eval": int(len(truth)),
            "positive_rate_eval": float(truth.mean()) if len(truth) else 0.0,
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(truth, predictions)),
            "precision": float(precision_score(truth, predictions, zero_division=0)),
            "recall": float(recall_score(truth, predictions, zero_division=0)),
            "brier_score": float(brier_score_loss(truth, probabilities)),
            "log_loss": float(log_loss(truth, probabilities, labels=[0, 1])),
        }
        metrics["roc_auc"] = float(roc_auc_score(truth, probabilities)) if truth.nunique() > 1 else None
        return metrics


class ChronosForecaster:
    def __init__(self, enabled: bool, model_id: str, device: str):
        self.enabled = enabled
        self.model_id = model_id
        self.device = device
        self.pipeline = None

    def load(self) -> bool:
        if not self.enabled:
            return False
        if self.pipeline is not None:
            return True
        try:
            from chronos import Chronos2Pipeline
        except ImportError:
            logger.warning("Chronos is enabled but chronos-forecasting is not installed")
            return False

        self.pipeline = Chronos2Pipeline.from_pretrained(self.model_id, device_map=self.device)
        return True

    def forecast_direction(
        self,
        symbol: str,
        frame: pd.DataFrame,
        timeframe: str,
        prediction_length: int,
    ) -> float | None:
        if not self.load():
            return None
        if frame.empty:
            return None

        frequency = pandas_frequency(timeframe)
        context = frame[["time", "close"]].copy()
        context.rename(columns={"time": "timestamp", "close": "target"}, inplace=True)
        context["id"] = symbol
        context = context.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        if len(context) < 3:
            return None
        last_timestamp = pd.Timestamp(context["timestamp"].iloc[-1])
        if last_timestamp.tzinfo is not None:
            last_timestamp = last_timestamp.tz_convert("UTC").tz_localize(None)
        context["timestamp"] = pd.date_range(
            end=last_timestamp,
            periods=len(context),
            freq=frequency,
        )
        future_index = pd.date_range(
            start=last_timestamp + pd.Timedelta(frequency),
            periods=prediction_length,
            freq=frequency,
        )
        future_df = pd.DataFrame({"id": symbol, "timestamp": future_index})
        prediction = self.pipeline.predict_df(
            df=context[["id", "timestamp", "target"]],
            future_df=future_df,
            prediction_length=prediction_length,
            quantile_levels=[0.5],
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )
        if prediction is None or len(prediction) == 0:
            return None

        last_close = float(context["target"].iloc[-1])
        for column in ("predictions", "prediction", "median", "mean", "0.5"):
            if column not in prediction.columns:
                continue
            value = prediction[column].iloc[-1]
            if isinstance(value, (list, tuple, np.ndarray)):
                predicted = float(np.asarray(value).reshape(-1)[-1])
            else:
                predicted = float(value)
            return (predicted - last_close) / last_close

        logger.warning("Chronos output did not contain a supported prediction column")
        return None
