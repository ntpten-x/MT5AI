from __future__ import annotations

import pandas as pd


def split_time_series_frame(
    frame: pd.DataFrame,
    train_ratio: float,
    calibration_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = len(frame)
    if total < 3:
        raise ValueError("Need at least 3 rows to split time series frame")

    train_end = max(1, int(total * train_ratio))
    calibration_end = max(train_end + 1, int(total * (train_ratio + calibration_ratio)))
    calibration_end = min(calibration_end, total - 1)

    if train_end >= total - 1:
        train_end = max(1, total - 2)
    if calibration_end <= train_end:
        calibration_end = min(total - 1, train_end + max(1, (total - train_end) // 2))

    train = frame.iloc[:train_end].copy()
    calibration = frame.iloc[train_end:calibration_end].copy()
    test = frame.iloc[calibration_end:].copy()
    return train, calibration, test


def target_frame(frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    copy = frame.copy()
    copy["target"] = copy[target_column].astype(int)
    return copy


def split_train_calibration_frame(
    frame: pd.DataFrame,
    calibration_share: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total = len(frame)
    if total < 2:
        raise ValueError("Need at least 2 rows to split train and calibration")

    calibration_rows = max(1, int(total * calibration_share))
    if calibration_rows >= total:
        calibration_rows = 1

    split_index = total - calibration_rows
    if split_index <= 0:
        split_index = 1

    train = frame.iloc[:split_index].copy()
    calibration = frame.iloc[split_index:].copy()
    return train, calibration
