"""
Визуализация результатов обучения, градиентов и аномалий.
Создает набор графиков для анализа.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, Optional
import os


class ResultsVisualizer:
    """Визуализация результатов обучения и анализа градиентов."""

    def __init__(self, save_dir: str = ".", figsize: tuple = (16, 12)):
        self.save_dir = save_dir
        self.figsize = figsize
        os.makedirs(save_dir, exist_ok=True)

    def plot_all(
        self,
        history: Dict,
        title: str = "Training Results",
        anomaly_events: Optional[list] = None,
    ) -> str:
        """
        Создаёт комплексный дашборд со всеми графиками.

        Returns:
            Путь к сохранённому файлу
        """
        fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

        # 1. Loss curves
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_loss(history, ax1)

        # 2. Accuracy curve
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_accuracy(history, ax2)

        # 3. Gradient L2 norm
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_gradient_norm(history, ax3, anomaly_events)

        # 4. Gradient distribution
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_gradient_distribution(history, ax4)

        # 5. Anomaly timeline
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_anomaly_timeline(history, ax5, anomaly_events)

        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
        filepath = os.path.join(self.save_dir, "results_dashboard.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Dashboard saved: {filepath}")
        return filepath

    def _plot_loss(self, history: Dict, ax: plt.Axes):
        """График функции потерь."""
        train_loss = history["train_loss"]
        test_loss = history["test_loss"]
        epochs = range(1, len(train_loss) + 1)

        ax.plot(epochs, train_loss, "b-", linewidth=1.5, label="Train loss")
        ax.plot(epochs, test_loss, "r-", linewidth=1.5, label="Test loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_accuracy(self, history: Dict, ax: plt.Axes):
        """График точности."""
        accuracy = history["test_accuracy"]
        epochs = range(1, len(accuracy) + 1)

        ax.plot(epochs, accuracy, "g-", linewidth=1.5)
        ax.axhline(y=max(accuracy), color="g", linestyle="--", alpha=0.5,
                    label=f"Best: {max(accuracy):.4f}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Test Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_gradient_norm(
        self, history: Dict, ax: plt.Axes,
        anomaly_events: Optional[list] = None
    ):
        """График L2-нормы градиентов."""
        l2_norms = history["gradient_l2_norm"]
        epochs = range(1, len(l2_norms) + 1)

        ax.plot(epochs, l2_norms, "purple", linewidth=1.5, label="L2 norm")
        ax.fill_between(epochs, l2_norms, alpha=0.2, color="purple")

        # Среднее
        mean_l2 = np.mean(l2_norms)
        ax.axhline(y=mean_l2, color="orange", linestyle="--", alpha=0.7,
                    label=f"Mean: {mean_l2:.4f}")

        # Аномалии
        if anomaly_events:
            anom_epochs = [e["epoch"] for e in anomaly_events if e.get("anomaly")]
            for ep in anom_epochs:
                if 1 <= ep <= len(l2_norms):
                    ax.axvline(x=ep, color="red", linestyle=":", alpha=0.5)
                    ax.annotate("", xy=(ep, mean_l2), xytext=(ep, mean_l2 + 0.3),
                               arrowprops=dict(arrowstyle="->", color="red", alpha=0.7))

        ax.set_xlabel("Epoch")
        ax.set_ylabel("L2 Norm")
        ax.set_title("Gradient L2 Norm Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_gradient_distribution(self, history: Dict, ax: plt.Axes):
        """Гистограмма распределения L2-норм."""
        l2_norms = np.array(history["gradient_l2_norm"])

        ax.hist(l2_norms, bins=15, color="steelblue", edgecolor="white",
                alpha=0.8, density=True)
        ax.axvline(x=l2_norms.mean(), color="red", linestyle="--",
                    label=f"Mean: {l2_norms.mean():.4f}")
        ax.axvline(x=l2_norms.mean() + l2_norms.std(), color="orange",
                    linestyle="--", label=f"+1σ: {l2_norms.mean() + l2_norms.std():.4f}")

        ax.set_xlabel("L2 Norm")
        ax.set_ylabel("Density")
        ax.set_title("Gradient L2 Norm Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_anomaly_timeline(
        self, history: Dict, ax: plt.Axes,
        anomaly_events: Optional[list] = None
    ):
        """Временная шкала событий градиентов."""
        l2_norms = np.array(history["gradient_l2_norm"])
        epochs = np.arange(1, len(l2_norms) + 1)

        # Нормализуем для отображения
        l2_normalized = (l2_norms - l2_norms.min()) / (l2_norms.max() - l2_norms.min() + 1e-8)

        ax.plot(epochs, l2_normalized, "k-", alpha=0.3, linewidth=0.8)

        # События
        clipped_events = [e for e in history.get("gradient_events", []) if e.get("clipped")]
        vanishing_events = [e for e in history.get("gradient_events", []) if e.get("vanishing")]
        anomaly_events_list = [e for e in history.get("gradient_events", []) if e.get("anomaly")]

        # Clipped
        if clipped_events:
            clipped_epochs = [e["epoch"] for e in clipped_events]
            clipped_l2 = [l2_norms[e - 1] for e in clipped_epochs if 0 < e <= len(l2_norms)]
            if clipped_l2:
                clipped_norm = [(l - l2_norms.min()) / (l2_norms.max() - l2_norms.min() + 1e-8)
                               for l in clipped_l2]
                ax.scatter(clipped_epochs, clipped_norm, c="orange", s=50,
                          marker="s", label="Clipped", zorder=5)

        # Anomalies
        if anomaly_events_list:
            anom_epochs = [e["epoch"] for e in anomaly_events_list]
            anom_l2 = [l2_norms[e - 1] for e in anom_epochs if 0 < e <= len(l2_norms)]
            if anom_l2:
                anom_norm = [(l - l2_norms.min()) / (l2_norms.max() - l2_norms.min() + 1e-8)
                            for l in anom_l2]
                ax.scatter(anom_epochs, anom_norm, c="red", s=80,
                          marker="*", label="Anomaly", zorder=6)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Normalized L2 Norm")
        ax.set_title("Gradient Events Timeline")
        ax.legend()
        ax.grid(True, alpha=0.3)

    @staticmethod
    def plot_comparison(
        history_clean: Dict,
        history_poisoned: Dict,
        title: str = "Clean vs Poisoned Training",
    ) -> str:
        """Сравнение обучения на чистых и отравленных данных."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss
        ax1 = axes[0, 0]
        ax1.plot(history_clean["train_loss"], "b-", label="Clean Train")
        ax1.plot(history_clean["test_loss"], "b--", label="Clean Test")
        ax1.plot(history_poisoned["train_loss"], "r-", label="Poisoned Train")
        ax1.plot(history_poisoned["test_loss"], "r--", label="Poisoned Test")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss Comparison")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy
        ax2 = axes[0, 1]
        ax2.plot(history_clean["test_accuracy"], "b-", label="Clean")
        ax2.plot(history_poisoned["test_accuracy"], "r-", label="Poisoned")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Test Accuracy Comparison")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Gradient L2
        ax3 = axes[1, 0]
        ax3.plot(history_clean["gradient_l2_norm"], "b-", label="Clean")
        ax3.plot(history_poisoned["gradient_l2_norm"], "r-", label="Poisoned")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("L2 Norm")
        ax3.set_title("Gradient L2 Norm Comparison")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Summary stats
        ax4 = axes[1, 1]
        ax4.axis("off")
        metrics = [
            f"Clean  - Final Acc: {history_clean['test_accuracy'][-1]:.4f}",
            f"Clean  - Best Acc:  {max(history_clean['test_accuracy']):.4f}",
            f"Clean  - Mean L2:   {np.mean(history_clean['gradient_l2_norm']):.4f}",
            f"Poison - Final Acc: {history_poisoned['test_accuracy'][-1]:.4f}",
            f"Poison - Best Acc:  {max(history_poisoned['test_accuracy']):.4f}",
            f"Poison - Mean L2:   {np.mean(history_poisoned['gradient_l2_norm']):.4f}",
        ]
        ax4.text(0.1, 0.5, "\n".join(metrics), fontsize=11,
                family="monospace", va="center")
        ax4.set_title("Summary Statistics")

        fig.suptitle(title, fontsize=14, fontweight="bold")
        filepath = os.path.join("images", "comparison.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Comparison saved: {filepath}")
        return filepath


def run_visualization():
    """Полный цикл: обучение + визуализация."""
    from data_generator import generate_data
    from neural_network import (
        BinaryClassifier, GradientAnalyzer, prepare_tensors,
        train_model as train_baseline,
    )
    from attacks import label_flipping
    from gradient_control import TrainedWithGradientControl

    X_train, X_test, y_train, y_test = generate_data()
    input_dim = X_train.shape[1]
    device = "cpu"

    X_t, X_te, y_t, y_te = prepare_tensors(X_train, X_test, y_train, y_test, device)

    # 1. Baseline training
    print("\n--- Baseline Training ---")
    model_base = BinaryClassifier(input_dim=input_dim, hidden_dim=64)
    history_base = train_baseline(model_base, X_t, y_t, X_te, y_te, epochs=40, verbose=10)

    # 2. Poisoned training
    _, y_train_poisoned, _, _ = label_flipping(y_train, y_test, flip_rate=0.15)
    X_t_poison, X_te_poison, y_t_poison, y_te_poison = prepare_tensors(
        X_train, X_test, y_train_poisoned, y_test, device
    )

    print("\n--- Poisoned Training ---")
    model_poison = BinaryClassifier(input_dim=input_dim, hidden_dim=64)
    history_poison = train_baseline(model_poison, X_t_poison, y_t_poison, X_te_poison, y_te_poison,
                                     epochs=40, verbose=10)

    # 3. Controlled training
    print("\n--- Controlled Training ---")
    model_control = BinaryClassifier(input_dim=input_dim, hidden_dim=64)
    trainer = TrainedWithGradientControl(model=model_control, max_norm=1.0)
    history_control = trainer.train(
        X_t, y_t, X_te, y_te, epochs=40, lr=0.001, batch_size=64, verbose=5
    )

    # 4. Visualization
    print("\n--- Visualization ---")
    viz = ResultsVisualizer(save_dir="images")

    # Dashboard for controlled training
    viz.plot_all(
        history_control,
        title="Gradient-Controlled Training Dashboard",
        anomaly_events=history_control.get("gradient_events", []),
    )

    # Comparison
    ResultsVisualizer.plot_comparison(history_base, history_poisoned=history_poison)

    print("\nDone! All visualizations saved.")


if __name__ == "__main__":
    run_visualization()
