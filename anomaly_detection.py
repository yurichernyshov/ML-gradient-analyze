"""
Методы обнаружения аномалий для градиентов и предсказаний модели.
- IQR (Inter-Quartile Range)
- Z-Score
"""

import numpy as np
from typing import Tuple, Optional


class IQRDetector:
    """Обнаружение аномалий методом межквартильного размаха (IQR)."""

    def __init__(self, multiplier: float = 1.5):
        """
        Args:
            multiplier: Множитель для IQR (1.5 — обычные выбросы, 3 — экстремальные)
        """
        self.multiplier = multiplier
        self.q1 = None
        self.q3 = None
        self.iqr = None
        self.is_fitted = False

    def fit(self, data: np.ndarray) -> "IQRDetector":
        """Вычисляет квартили на обучающих данных."""
        data = np.asarray(data).flatten()
        self.q1 = np.percentile(data, 25)
        self.q3 = np.percentile(data, 75)
        self.iqr = self.q3 - self.q1
        self.is_fitted = True
        return self

    def detect(self, data: np.ndarray) -> np.ndarray:
        """
        Возвращает булев массив: True для аномалий, False для нормальных.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before detect()")
        data = np.asarray(data).flatten()
        lower = self.q1 - self.multiplier * self.iqr
        upper = self.q3 + self.multiplier * self.iqr
        return (data < lower) | (data > upper)

    def get_bounds(self) -> Tuple[float, float]:
        """Возвращает допустимый диапазон [lower, upper]."""
        lower = self.q1 - self.multiplier * self.iqr
        upper = self.q3 + self.multiplier * self.iqr
        return float(lower), float(upper)

    def summary(self, data: np.ndarray) -> dict:
        """Возвращает сводку по обнаруженным аномалиям."""
        anomalies = self.detect(data)
        return {
            "total": len(data),
            "anomalies": int(anomalies.sum()),
            "normal": int((~anomalies).sum()),
            "anomaly_rate": float(anomalies.mean()),
            "q1": self.q1,
            "q3": self.q3,
            "iqr": self.iqr,
            "lower_bound": self.q1 - self.multiplier * self.iqr,
            "upper_bound": self.q3 + self.multiplier * self.iqr,
        }


class ZScoreDetector:
    """Обнаружение аномалий методом Z-Score."""

    def __init__(self, threshold: float = 2.0):
        """
        Args:
            threshold: Порог Z-оценки (2.0 ≈ 95%, 3.0 ≈ 99.7%)
        """
        self.threshold = threshold
        self.mean = None
        self.std = None
        self.is_fitted = False

    def fit(self, data: np.ndarray) -> "ZScoreDetector":
        """Вычисляет среднее и стандартное отклонение."""
        data = np.asarray(data).flatten()
        self.mean = np.mean(data)
        self.std = np.std(data)
        if self.std == 0:
            self.std = 1e-8
        self.is_fitted = True
        return self

    def detect(self, data: np.ndarray) -> np.ndarray:
        """Возвращает булев массив: True для аномалий."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before detect()")
        data = np.asarray(data).flatten()
        z_scores = np.abs((data - self.mean) / self.std)
        return z_scores > self.threshold

    def get_z_scores(self, data: np.ndarray) -> np.ndarray:
        """Вычисляет Z-оценки для данных."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before detect()")
        data = np.asarray(data).flatten()
        return (data - self.mean) / self.std

    def summary(self, data: np.ndarray) -> dict:
        """Возвращает сводку по обнаруженным аномалиям."""
        anomalies = self.detect(data)
        z_scores = self.get_z_scores(data)
        return {
            "total": len(data),
            "anomalies": int(anomalies.sum()),
            "normal": int((~anomalies).sum()),
            "anomaly_rate": float(anomalies.mean()),
            "threshold": self.threshold,
            "mean": self.mean,
            "std": self.std,
            "z_mean": float(np.mean(z_scores)),
            "z_std": float(np.std(z_scores)),
            "z_max": float(np.max(np.abs(z_scores))),
        }


def compare_detectors(
    clean_data: np.ndarray,
    poisoned_data: np.ndarray,
) -> Tuple[dict, dict]:
    """
    Сравнивает IQR и Z-Score на чистых и отравленных данных.

    Returns:
        (iqr_summary, zscore_summary) для poisoned_data
    """
    # Обучаем на чистых данных
    iqr = IQRDetector(multiplier=1.5).fit(clean_data)
    zscore = ZScoreDetector(threshold=2.0).fit(clean_data)

    print("=" * 60)
    print("Обнаружение аномалий: IQR vs Z-Score")
    print("=" * 60)

    print("\n--- IQR Detector ---")
    print(f"Clean data stats: {IQRDetector(multiplier=1.5).fit(clean_data).summary(clean_data)}")
    iqr_poisoned = iqr.summary(poisoned_data)
    print(f"Poisoned data anomalies: {iqr_poisoned}")

    print("\n--- Z-Score Detector ---")
    print(f"Clean data stats: {ZScoreDetector(threshold=2.0).fit(clean_data).summary(clean_data)}")
    zscore_poisoned = zscore.summary(poisoned_data)
    print(f"Poisoned data anomalies: {zscore_poisoned}")

    return iqr_poisoned, zscore_poisoned


if __name__ == "__main__":
    from data_generator import generate_data
    from attacks import label_flipping
    from neural_network import train_model, BinaryClassifier, GradientAnalyzer, prepare_tensors

    X_train, X_test, y_train, y_test = generate_data()
    input_dim = X_train.shape[1]
    device = "cpu"

    # Обучаем модель на чистых данных
    model = BinaryClassifier(input_dim=input_dim, hidden_dim=64)
    X_t, X_te, y_t, y_te = prepare_tensors(X_train, X_test, y_train, y_test, device)

    print("\n--- Обучение на чистых данных ---")
    history_clean = train_model(model, X_t, y_t, X_te, y_te, epochs=30, verbose=10)
    clean_l2_norms = history_clean["gradient_l2_norm"]

    # Получаем отравленные данные
    _, y_train_poisoned, _, _ = label_flipping(y_train, y_test, flip_rate=0.15)

    # Обучаем на отравленных данных
    model_poisoned = BinaryClassifier(input_dim=input_dim, hidden_dim=64)
    X_t_poison, X_te_poison, y_t_poison, y_te_poison = prepare_tensors(
        X_train, X_test, y_train_poisoned, y_test, device
    )

    print("\n--- Обучение на отравленных данных ---")
    history_poisoned = train_model(model_poisoned, X_t_poison, y_t_poison, X_te_poison, y_te_poison,
                                    epochs=30, verbose=10)
    poisoned_l2_norms = history_poisoned["gradient_l2_norm"]

    # Применяем детекторы
    iqr_result, zscore_result = compare_detectors(clean_l2_norms, poisoned_l2_norms)
