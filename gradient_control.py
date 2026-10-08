"""
Обучение нейронной сети с контролем градиентов на каждом шаге.
Включает:
- Gradient Clipping (по норме и по значению)
- Gradient Monitoring (отслеживание взрывов/затухания)
- Anomaly-aware training (остановка при обнаружении аномалий)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Optional, List, Dict
from neural_network import BinaryClassifier, GradientAnalyzer, prepare_tensors
from anomaly_detection import IQRDetector, ZScoreDetector


class GradientController:
    """Контроль градиентов во время обучения."""

    def __init__(
        self,
        max_norm: Optional[float] = 1.0,
        min_norm: float = 1e-7,
        use_anomaly_detection: bool = True,
        anomaly_threshold_z: float = 2.5,
        anomaly_multiplier_iqr: float = 1.5,
    ):
        """
        Args:
            max_norm: Максимальная L2-норма градиентов (clipping)
            min_norm: Минимальная норма (детекция затухания)
            use_anomaly_detection: Включить аномальное обнаружение
            anomaly_threshold_z: Порог Z-Score
            anomaly_multiplier_iqr: Множитель IQR
        """
        self.max_norm = max_norm
        self.min_norm = min_norm
        self.use_anomaly_detection = use_anomaly_detection
        self.anomaly_threshold_z = anomaly_threshold_z
        self.anomaly_multiplier_iqr = anomaly_multiplier_iqr

        # Для мониторинга
        self.l2_history: List[float] = []
        self.gradient_events: List[Dict] = []

    def clip_gradients(self, model: nn.Module) -> Tuple[bool, str]:
        """
        Применяет gradient clipping.

        Returns:
            (warning_triggered, message)
        """
        warnings = []

        # L2-clip
        if self.max_norm is not None:
            grad_norm = GradientAnalyzer.compute_l2_norm(model)
            if grad_norm > self.max_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
                warnings.append(f"Grad norm clipped: {grad_norm:.4f} -> {self.max_norm:.4f}")

        return len(warnings) > 0, "; ".join(warnings)

    def check_min_norm(self, model: nn.Module) -> Tuple[bool, str]:
        """Проверяет, не затухли ли градиенты."""
        grad_norm = GradientAnalyzer.compute_l2_norm(model)
        if grad_norm < self.min_norm:
            return True, f"Gradient vanishing detected: {grad_norm:.2e}"
        return False, ""

    def check_anomaly(self, current_l2: float) -> Tuple[bool, str]:
        """Проверяет аномальность текущего значения нормы градиентов."""
        if not self.use_anomaly_detection or len(self.l2_history) < 5:
            return False, ""

        l2_array = np.array(self.l2_history)

        # Z-Score
        z_detector = ZScoreDetector(threshold=self.anomaly_threshold_z).fit(l2_array)
        if z_detector.detect(np.array([current_l2]))[0]:
            z_scores = z_detector.get_z_scores(np.array([current_l2]))
            return True, f"Z-Score anomaly: z={z_scores[0]:.2f}"

        # IQR
        iqr_detector = IQRDetector(multiplier=self.anomaly_multiplier_iqr).fit(l2_array)
        if iqr_detector.detect(np.array([current_l2]))[0]:
            return True, "IQR anomaly detected"

        return False, ""

    def record(self, model: nn.Module, epoch: int, batch_idx: int = 0) -> Dict:
        """
        Записывает и проверяет градиенты после шага обучения.

        Returns:
            Словарь с результатами проверки
        """
        l2_norm = GradientAnalyzer.compute_l2_norm(model)
        self.l2_history.append(l2_norm)

        result = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "l2_norm": l2_norm,
            "clipped": False,
            "vanishing": False,
            "anomaly": False,
            "message": "",
        }

        # Проверка clipping
        if self.max_norm is not None:
            has_warning, msg = self.clip_gradients(model)
            result["clipped"] = has_warning
            result["message"] += msg + "; " if msg else ""

        # Проверка затухания
        is_vanishing, msg = self.check_min_norm(model)
        result["vanishing"] = is_vanishing
        result["message"] += msg + "; " if msg else ""

        # Проверка аномалий
        is_anomaly, msg = self.check_anomaly(l2_norm)
        result["anomaly"] = is_anomaly
        result["message"] += msg if msg else ""

        if result["message"]:
            self.gradient_events.append(result)

        return result


class TrainedWithGradientControl:
    """
    Обучение с полным контролем градиентов.
    """

    def __init__(
        self,
        model: nn.Module,
        max_norm: float = 1.0,
        min_norm: float = 1e-7,
        stop_on_anomaly: bool = False,
        anomaly_warmup: int = 10,
    ):
        """
        Args:
            model: Модель для обучения
            max_norm: Максимальная L2-норма градиентов
            min_norm: Минимальная норма (детекция затухания)
            stop_on_anomaly: Останавливать ли обучение при аномалии
            anomaly_warmup: Эпохи для накопления статистики перед обнаружением
        """
        self.model = model
        self.controller = GradientController(
            max_norm=max_norm,
            min_norm=min_norm,
            use_anomaly_detection=True,
        )
        self.stop_on_anomaly = stop_on_anomaly
        self.anomaly_warmup = anomaly_warmup
        self.training_stopped = False

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 64,
        verbose: int = 5,
    ) -> Dict:
        """
        Обучает модель с контролем градиентов.

        Returns:
            Словарь с полной историей обучения
        """
        device = next(self.model.parameters()).device
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        dataset = TensorDataset(X_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        history = {
            "train_loss": [],
            "test_loss": [],
            "test_accuracy": [],
            "gradient_l2_norm": [],
            "gradient_events": [],
            "stopped_epoch": None,
        }

        for epoch in range(1, epochs + 1):
            if self.training_stopped:
                print(f"Training stopped at epoch {epoch - 1}")
                break

            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            epoch_events = []

            for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # Контроль градиентов
                grad_result = self.controller.record(self.model, epoch, batch_idx)

                # Проверка остановки
                if self.stop_on_anomaly and grad_result["anomaly"]:
                    if epoch > self.anomaly_warmup:
                        self.training_stopped = True
                        history["stopped_epoch"] = epoch
                        print(f"\n⚠️  Training STOPPED at epoch {epoch} due to anomaly!")
                        print(f"   {grad_result['message']}")
                        break

                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
                epoch_events.append(grad_result)

            # Запись истории
            avg_train_loss = epoch_loss / n_batches if n_batches > 0 else 0

            self.model.eval()
            with torch.no_grad():
                test_outputs = self.model(X_test)
                test_loss = criterion(test_outputs, y_test).item()
                test_preds = (test_outputs > 0.5).float()
                accuracy = (test_preds == y_test).float().mean().item()

            l2_norm = self.controller.l2_history[-1] if self.controller.l2_history else 0

            history["train_loss"].append(avg_train_loss)
            history["test_loss"].append(test_loss)
            history["test_accuracy"].append(accuracy)
            history["gradient_l2_norm"].append(l2_norm)
            history["gradient_events"].extend(epoch_events)

            # Вывод
            event_str = ""
            if epoch_events:
                anomalous = [e for e in epoch_events if e["anomaly"] or e["clipped"] or e["vanishing"]]
                if anomalous:
                    event_str = f" | Events: {len(anomalous)} warnings"

            if verbose > 0 and (epoch % verbose == 0 or epoch == 1):
                print(f"Epoch {epoch:4d}/{epochs} | "
                      f"Train: {avg_train_loss:.4f} | "
                      f"Test: {test_loss:.4f} | "
                      f"Acc: {accuracy:.4f} | "
                      f"L2: {l2_norm:.4f}{event_str}")

        # Сводка
        self._print_summary(history)

        return history

    def _print_summary(self, history: Dict):
        """Выводит сводку по обучению."""
        print("\n" + "=" * 60)
        print("Сводка обучения с контролем градиентов")
        print("=" * 60)

        l2_norms = np.array(history["gradient_l2_norm"])
        print(f"\nГрадиенты:")
        print(f"  Среднее L2:  {l2_norms.mean():.6f}")
        print(f"  Мин L2:      {l2_norms.min():.6f}")
        print(f"  Макс L2:     {l2_norms.max():.6f}")
        print(f"  Std L2:      {l2_norms.std():.6f}")

        events = history.get("gradient_events", [])
        clipped = sum(1 for e in events if e.get("clipped"))
        vanishing = sum(1 for e in events if e.get("vanishing"))
        anomalies = sum(1 for e in events if e.get("anomaly"))

        print(f"\nСобытия градиентов:")
        print(f"  Clipped:     {clipped}")
        print(f"  Vanishing:   {vanishing}")
        print(f"  Anomalies:   {anomalies}")

        if history.get("stopped_epoch"):
            print(f"\n⚠️  Обучение остановлено на эпохе {history['stopped_epoch']}")

        print(f"\nЛучшая точность: {max(history['test_accuracy']):.4f}")
        print(f"Финальная точность: {history['test_accuracy'][-1]:.4f}")


def run_controlled_training():
    """Полный цикл обучения с контролем градиентов."""
    from data_generator import generate_data

    X_train, X_test, y_train, y_test = generate_data()
    input_dim = X_train.shape[1]
    device = "cpu"

    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ С КОНТРОЛЕМ ГРАДИЕНТОВ")
    print("=" * 60)

    # Без контроля (baseline)
    print("\n--- Baseline (без контроля) ---")
    model_baseline = BinaryClassifier(input_dim=input_dim, hidden_dim=64)
    X_t, X_te, y_t, y_te = prepare_tensors(X_train, X_test, y_train, y_test, device)
    from neural_network import train_model as train_baseline
    history_base = train_baseline(model_baseline, X_t, y_t, X_te, y_te, epochs=40, verbose=10)

    # С контролем
    print("\n--- С контролем градиентов ---")
    model_controlled = BinaryClassifier(input_dim=input_dim, hidden_dim=64)
    trainer = TrainedWithGradientControl(
        model=model_controlled,
        max_norm=1.0,
        min_norm=1e-7,
        stop_on_anomaly=False,
    )
    history_controlled = trainer.train(
        X_t, y_t, X_te, y_te,
        epochs=40,
        lr=0.001,
        batch_size=64,
        verbose=5,
    )

    # Сравнение
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ")
    print("=" * 60)
    print(f"{'Метрика':<25} {'Baseline':>12} {'Controlled':>12}")
    print("-" * 50)
    print(f"{'Final Accuracy':<25} {history_base['test_accuracy'][-1]:>12.4f} {history_controlled['test_accuracy'][-1]:>12.4f}")
    print(f"{'Best Accuracy':<25} {max(history_base['test_accuracy']):>12.4f} {max(history_controlled['test_accuracy']):>12.4f}")
    print(f"{'Mean Grad L2':<25} {np.mean(history_base['gradient_l2_norm']):>12.6f} {np.mean(history_controlled['gradient_l2_norm']):>12.6f}")
    print(f"{'Max Grad L2':<25} {np.max(history_base['gradient_l2_norm']):>12.6f} {np.max(history_controlled['gradient_l2_norm']):>12.6f}")


if __name__ == "__main__":
    run_controlled_training()
