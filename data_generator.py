"""
Генератор данных для бинарной классификации.
Использует sklearn.datasets.make_classification для создания
разделяемых классов с возможностью настройки параметров.
"""

import numpy as np
from sklearn.datasets import make_classification
from typing import Tuple


def generate_data(
    n_samples: int = 1000,
    n_features: int = 10,
    n_informative: int = 8,
    n_redundant: int = 2,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Tuple:
    """
    Генерирует датасет для бинарной классификации.

    Args:
        n_samples: Количество образцов
        n_features: Общее количество признаков
        n_informative: Количество информативных признаков
        n_redundant: Количество избыточных признаков
        random_state: Seed для воспроизводимости
        test_size: Доля тестовой выборки

    Returns:
        Кортеж (X_train, X_test, y_train, y_test)
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=2,
        random_state=random_state,
        flip_y=0.0,  # без начального переключения меток
    )

    # Разделение на train/test
    n_train = int(n_samples * (1 - test_size))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    print(f"Сгенерировано данных: {n_samples} образцов, {n_features} признаков")
    print(f"Train: {X_train.shape[0]} образцов, Test: {X_test.shape[0]} образцов")
    print(f"Баланс классов (train): {np.bincount(y_train)}")
    print(f"Баланс классов (test): {np.bincount(y_test)}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = generate_data()
    print(f"\nФормы данных:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test: {y_test.shape}")
