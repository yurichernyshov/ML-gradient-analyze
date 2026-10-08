"""
Методы атак на модель бинарной классификации.
- Label Flipping (инверсия меток)
- Poisoning attack (отравление обучающей выборки)
"""

import numpy as np
from typing import Tuple


def label_flipping(
    y_train: np.ndarray,
    y_test: np.ndarray,
    flip_rate: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Атака инверсии меток (Label Flipping).
    Случайным образом инвертирует часть меток в обучающей и тестовой выборках.

    Args:
        y_train: Метки обучающей выборки
        y_test: Метки тестовой выборки
        flip_rate: Доля инвертированных меток (0..1)
        random_state: Seed для воспроизводимости

    Returns:
        (y_train_clean, y_train_poisoned, y_test_clean, y_test_poisoned)
    """
    rng = np.random.RandomState(random_state)

    # Обучающая выборка
    y_train_clean = y_train.copy()
    y_train_poisoned = y_train.copy()
    n_flip_train = int(len(y_train) * flip_rate)
    flip_indices_train = rng.choice(len(y_train), n_flip_train, replace=False)
    y_train_poisoned[flip_indices_train] = 1 - y_train_poisoned[flip_indices_train]

    # Тестовая выборка
    y_test_clean = y_test.copy()
    y_test_poisoned = y_test.copy()
    n_flip_test = int(len(y_test) * flip_rate)
    flip_indices_test = rng.choice(len(y_test), n_flip_test, replace=False)
    y_test_poisoned[flip_indices_test] = 1 - y_test_poisoned[flip_indices_test]

    print(f"\nLabel Flipping Attack (rate={flip_rate:.1%}):")
    print(f"  Train flipped: {n_flip_train}/{len(y_train)} ({n_flip_train/len(y_train):.1%})")
    print(f"  Test flipped:  {n_flip_test}/{len(y_test)} ({n_flip_test/len(y_test):.1%})")
    print(f"  Train clean class balance: {np.bincount(y_train_clean)}")
    print(f"  Train poisoned class balance: {np.bincount(y_train_poisoned)}")
    print(f"  Test clean class balance: {np.bincount(y_test_clean)}")
    print(f"  Test poisoned class balance: {np.bincount(y_test_poisoned)}")

    return y_train_clean, y_train_poisoned, y_test_clean, y_test_poisoned


def targeted_poisoning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_class: int = 1,
    poison_rate: float = 0.1,
    noise_std: float = 0.5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Целевое отравление обучающей выборки.
    Добавляет зашумлённые образцы, имитирующие целевой класс.

    Args:
        X_train: Обучающая выборка (features)
        y_train: Метки обучающей выборки
        X_test: Тестовая выборка (features)
        y_test: Метки тестовой выборки
        target_class: Целевой класс для имитации
        poison_rate: Доля отравляющих образцов от размера train
        noise_std: Стандартное отклонение шума
        random_state: Seed

    Returns:
        (X_train_clean, X_train_poisoned, y_train_clean, y_train_poisoned)
    """
    rng = np.random.RandomState(random_state)
    n_poison = int(len(X_train) * poison_rate)

    # Выбираем случайные образцы целевого класса
    target_indices = np.where(y_train == target_class)[0]
    if len(target_indices) == 0:
        raise ValueError(f"No samples of class {target_class} found in training data")

    selected = rng.choice(target_indices, min(n_poison, len(target_indices)), replace=False)
    base_samples = X_train[selected]

    # Добавляем шум
    noise = rng.normal(0, noise_std, size=base_samples.shape)
    poisoned_samples = base_samples + noise

    # Новые метки = целевой класс
    new_labels = np.full(poisoned_samples.shape[0], target_class, dtype=y_train.dtype)

    # Добавляем в выборку
    X_poisoned = np.vstack([X_train, poisoned_samples])
    y_poisoned = np.concatenate([y_train, new_labels])

    print(f"\nTargeted Poisoning Attack:")
    print(f"  Target class: {target_class}")
    print(f"  Poisoned samples: {n_poison}")
    print(f"  Train class balance (clean): {np.bincount(y_train)}")
    print(f"  Train class balance (poisoned): {np.bincount(y_poisoned)}")

    return X_train, y_train, X_poisoned, y_poisoned


if __name__ == "__main__":
    from data_generator import generate_data

    X_train, X_test, y_train, y_test = generate_data()

    # Label flipping
    _, y_train_flip, _, y_test_flip = label_flipping(y_train, y_test, flip_rate=0.1)

    # Targeted poisoning
    _, _, X_poison, y_poison = targeted_poisoning(
        X_train, y_train, X_test, y_test,
        target_class=1, poison_rate=0.1, noise_std=0.5
    )
