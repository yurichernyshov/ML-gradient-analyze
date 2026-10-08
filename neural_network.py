"""
Нейронная сеть для бинарной классификации на PyTorch.
Включает:
- Прямой проход (forward)
- Извлечение градиентов
- Расчёт L2-нормы вектора градиентов
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, List, Optional


class BinaryClassifier(nn.Module):
    """Нейронная сеть для бинарной классификации."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход по нейронной сети."""
        return self.network(x)


class GradientAnalyzer:
    """Анализ градиентов модели."""

    @staticmethod
    def extract_gradients(model: nn.Module) -> dict:
        """
        Извлекает градиенты всех параметров модели.

        Returns:
            Словарь {имя_параметра: градиент}
        """
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.detach().clone()
        return gradients

    @staticmethod
    def compute_l2_norm(model: nn.Module) -> float:
        """
        Вычисляет L2-норму вектора всех градиентов модели.

        Returns:
            L2-норма градиентов
        """
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        return float(np.sqrt(total_norm))

    @staticmethod
    def get_gradient_stats(model: nn.Module) -> dict:
        """
        Вычисляет статистику по градиентам.

        Returns:
            Словарь со статистикой градиентов
        """
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.data.abs())

        if not grads:
            return {}

        all_grads = torch.cat([g.flatten() for g in grads])
        return {
            "mean": all_grads.mean().item(),
            "std": all_grads.std().item(),
            "min": all_grads.min().item(),
            "max": all_grads.max().item(),
            "median": all_grads.median().item(),
            "l2_norm": GradientAnalyzer.compute_l2_norm(model),
        }


def prepare_tensors(X_train: np.ndarray, X_test: np.ndarray,
                    y_train: np.ndarray, y_test: np.ndarray,
                    device: str = "cpu") -> Tuple:
    """Преобразует данные в torch.Tensor."""
    X_train_t = torch.FloatTensor(X_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1).to(device)
    return X_train_t, X_test_t, y_train_t, y_test_t


def train_model(
    model: nn.Module,
    X_train: torch.Tensor, y_train: torch.Tensor,
    X_test: torch.Tensor, y_test: torch.Tensor,
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 64,
    device: str = "cpu",
    verbose: int = 10,
) -> dict:
    """
    Обучает модель бинарной классификации.

    Returns:
        Словарь с историей обучения
    """
    model = model.to(device)
    model.train()

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = {
        "train_loss": [],
        "test_loss": [],
        "test_accuracy": [],
        "gradient_l2_norm": [],
        "gradient_stats": [],
    }

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # Test
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test).item()
            test_preds = (test_outputs > 0.5).float()
            accuracy = (test_preds == y_test).float().mean().item()

        # Gradient analysis
        grad_l2 = GradientAnalyzer.compute_l2_norm(model)
        grad_stats = GradientAnalyzer.get_gradient_stats(model)

        history["train_loss"].append(avg_train_loss)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(accuracy)
        history["gradient_l2_norm"].append(grad_l2)
        history["gradient_stats"].append(grad_stats)

        if verbose > 0 and epoch % verbose == 0:
            print(f"Epoch {epoch:4d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | "
                  f"Test Acc: {accuracy:.4f} | "
                  f"Grad L2: {grad_l2:.6f}")

    return history


if __name__ == "__main__":
    from data_generator import generate_data

    X_train, X_test, y_train, y_test = generate_data()
    input_dim = X_train.shape[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = BinaryClassifier(input_dim=input_dim, hidden_dim=64)
    print(f"\nМодель:\n{model}")

    # Test forward pass
    X_t, _, y_t, _ = prepare_tensors(X_train, X_test, y_train, y_test, device)
    with torch.no_grad():
        pred = model(X_t[:5])
    print(f"\nПример прямого прохода (5 образцов): {pred.flatten().tolist()}")

    # Test gradient extraction
    criterion = nn.BCELoss()
    pred_grad = model(X_t[:5])
    loss = criterion(pred_grad, y_t[:5])
    loss.backward()

    grad = GradientAnalyzer.extract_gradients(model)
    print(f"\nИзвлечено градиентов: {len(grad)} параметров")
    for name, g in list(grad.items())[:3]:
        print(f"  {name}: grad shape={g.shape}, mean={g.abs().mean().item():.6f}")

    l2_norm = GradientAnalyzer.compute_l2_norm(model)
    print(f"\nL2-норма градиентов: {l2_norm:.6f}")

    stats = GradientAnalyzer.get_gradient_stats(model)
    print(f"\nСтатистика градиентов: {stats}")
