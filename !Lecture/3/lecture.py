#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Практичний ноутбук до Лекції 3: Навчання нейронних мереж
Курс: Deep Learning
"""

# ============================================================================
# ЧАСТИНА 0: НАЛАШТУВАННЯ
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from tqdm import tqdm

# Для візуалізації
from IPython.display import HTML
import matplotlib.animation as animation

import warnings

warnings.filterwarnings('ignore')

# Налаштування
plt.style.use('seaborn-v0_8-darkgrid')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Використовуємо device: {device}")

# Seed для відтворюваності
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# ЧАСТИНА 1: BACKPROPAGATION - РУЧНА РЕАЛІЗАЦІЯ
# ============================================================================

print("=" * 60)
print("ЧАСТИНА 1: BACKPROPAGATION КРОК ЗА КРОКОМ")
print("=" * 60)


class SimpleNetwork:
    """Проста 2-шарова мережа для демонстрації backprop"""

    def __init__(self, input_size=2, hidden_size=3, output_size=1):
        # Ініціалізація ваг
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))

        # Для збереження проміжних значень
        self.cache = {}

    def relu(self, z):
        """ReLU активація"""
        return np.maximum(0, z)

    def relu_derivative(self, z):
        """Похідна ReLU"""
        return (z > 0).astype(float)

    def forward(self, X):
        """
        Forward pass з збереженням всіх проміжних значень
        для backpropagation
        """
        # Шар 1
        self.cache['X'] = X
        self.cache['z1'] = X @ self.W1 + self.b1
        self.cache['a1'] = self.relu(self.cache['z1'])

        # Шар 2
        self.cache['z2'] = self.cache['a1'] @ self.W2 + self.b2
        self.cache['a2'] = self.cache['z2']  # Без активації на виході

        return self.cache['a2']

    def compute_loss(self, y_pred, y_true):
        """MSE loss"""
        m = y_true.shape[0]
        loss = np.mean((y_pred - y_true) ** 2)
        return loss

    def backward(self, y_true, learning_rate=0.01, verbose=False):
        """
        Backpropagation з детальним виведенням
        """
        m = y_true.shape[0]

        # Градієнт по виходу (MSE loss)
        dL_da2 = 2 * (self.cache['a2'] - y_true) / m

        if verbose:
            print("\n📊 BACKPROPAGATION КРОК ЗА КРОКОМ:")
            print(f"1. Градієнт втрат по виходу: dL/da2 shape = {dL_da2.shape}")

        # Градієнти для шару 2
        dL_dW2 = self.cache['a1'].T @ dL_da2
        dL_db2 = np.sum(dL_da2, axis=0, keepdims=True)

        if verbose:
            print(f"2. Градієнт по W2: dL/dW2 shape = {dL_dW2.shape}")
            print(f"   Градієнт по b2: dL/db2 shape = {dL_db2.shape}")

        # Градієнт через шар 2
        dL_da1 = dL_da2 @ self.W2.T

        if verbose:
            print(f"3. Градієнт пропагується на шар 1: dL/da1 shape = {dL_da1.shape}")

        # Градієнт через ReLU
        dL_dz1 = dL_da1 * self.relu_derivative(self.cache['z1'])

        if verbose:
            print(f"4. Градієнт через ReLU: dL/dz1 shape = {dL_dz1.shape}")

        # Градієнти для шару 1
        dL_dW1 = self.cache['X'].T @ dL_dz1
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

        if verbose:
            print(f"5. Градієнт по W1: dL/dW1 shape = {dL_dW1.shape}")
            print(f"   Градієнт по b1: dL/db1 shape = {dL_db1.shape}")

        # Оновлення ваг
        self.W2 -= learning_rate * dL_dW2
        self.b2 -= learning_rate * dL_db2
        self.W1 -= learning_rate * dL_dW1
        self.b1 -= learning_rate * dL_db1

        # Повертаємо градієнти для візуалізації
        return {
            'dW1': dL_dW1, 'db1': dL_db1,
            'dW2': dL_dW2, 'db2': dL_db2
        }


# Демонстрація backpropagation
def demo_backpropagation():
    """Покрокова демонстрація backprop"""

    print("\n🎯 Демонстрація Backpropagation на простому прикладі")

    # Створюємо просту задачу
    X = np.array([[0.5, 0.3], [0.2, 0.8], [0.9, 0.1], [0.4, 0.6]])
    y = np.array([[0.8], [1.0], [0.7], [0.9]])

    print(f"Вхідні дані: {X.shape}")
    print(f"Цільові значення: {y.shape}")

    # Створюємо мережу
    net = SimpleNetwork(input_size=2, hidden_size=3, output_size=1)

    # Forward pass
    print("\n➡️ FORWARD PASS:")
    y_pred = net.forward(X)
    loss = net.compute_loss(y_pred, y)
    print(f"Передбачення: {y_pred.flatten()}")
    print(f"Початкова втрата: {loss:.4f}")

    # Backward pass з детальним виведенням
    print("\n⬅️ BACKWARD PASS:")
    gradients = net.backward(y, learning_rate=0.1, verbose=True)

    # Показуємо величини градієнтів
    print("\n📈 Величини градієнтів:")
    for name, grad in gradients.items():
        print(f"{name}: mean={np.mean(np.abs(grad)):.6f}, max={np.max(np.abs(grad)):.6f}")

    # Новий forward pass після оновлення
    y_pred_new = net.forward(X)
    loss_new = net.compute_loss(y_pred_new, y)
    print(f"\n✅ Втрата після оновлення: {loss_new:.4f} (зменшення на {loss - loss_new:.4f})")


demo_backpropagation()

# ============================================================================
# ЧАСТИНА 2: ФУНКЦІЇ ВТРАТ - ПОРІВНЯННЯ
# ============================================================================

print("\n" + "=" * 60)
print("ЧАСТИНА 2: ФУНКЦІЇ ВТРАТ")
print("=" * 60)


def visualize_loss_functions():
    """Візуалізація різних функцій втрат"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. MSE для регресії
    y_true = 0
    y_pred = np.linspace(-3, 3, 100)
    mse = (y_pred - y_true) ** 2
    mse_grad = 2 * (y_pred - y_true)

    axes[0, 0].plot(y_pred, mse, 'b-', linewidth=2, label='MSE')
    axes[0, 0].plot(y_pred, mse_grad, 'r--', linewidth=2, label='Gradient')
    axes[0, 0].axvline(x=y_true, color='g', linestyle=':', label='True value')
    axes[0, 0].set_title('MSE Loss')
    axes[0, 0].set_xlabel('Prediction')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. MAE для регресії
    mae = np.abs(y_pred - y_true)
    mae_grad = np.sign(y_pred - y_true)

    axes[0, 1].plot(y_pred, mae, 'b-', linewidth=2, label='MAE')
    axes[0, 1].plot(y_pred, mae_grad, 'r--', linewidth=2, label='Gradient')
    axes[0, 1].axvline(x=y_true, color='g', linestyle=':', label='True value')
    axes[0, 1].set_title('MAE Loss')
    axes[0, 1].set_xlabel('Prediction')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Huber Loss
    delta = 1.0
    huber = np.where(np.abs(y_pred - y_true) <= delta,
                     0.5 * (y_pred - y_true) ** 2,
                     delta * (np.abs(y_pred - y_true) - 0.5 * delta))

    axes[0, 2].plot(y_pred, huber, 'b-', linewidth=2, label=f'Huber (δ={delta})')
    axes[0, 2].plot(y_pred, mse, 'g--', alpha=0.5, label='MSE')
    axes[0, 2].plot(y_pred, mae, 'r--', alpha=0.5, label='MAE')
    axes[0, 2].set_title('Huber Loss (MSE + MAE hybrid)')
    axes[0, 2].set_xlabel('Prediction')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Binary Cross-Entropy
    y_true_class = 1
    y_pred_prob = np.linspace(0.01, 0.99, 100)
    bce = -y_true_class * np.log(y_pred_prob) - (1 - y_true_class) * np.log(1 - y_pred_prob)
    bce_grad = -y_true_class / y_pred_prob + (1 - y_true_class) / (1 - y_pred_prob)

    axes[1, 0].plot(y_pred_prob, bce, 'b-', linewidth=2, label='BCE (y_true=1)')
    axes[1, 0].set_title('Binary Cross-Entropy')
    axes[1, 0].set_xlabel('Predicted Probability')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Порівняння BCE vs MSE для класифікації
    axes[1, 1].plot(y_pred_prob, bce, 'b-', linewidth=2, label='BCE')
    axes[1, 1].plot(y_pred_prob, (y_pred_prob - y_true_class) ** 2, 'r-', linewidth=2, label='MSE')
    axes[1, 1].set_title('BCE vs MSE для класифікації')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Focal Loss (для imbalanced datasets)
    gamma = 2
    focal = -y_true_class * (1 - y_pred_prob) ** gamma * np.log(y_pred_prob)

    axes[1, 2].plot(y_pred_prob, bce, 'b-', linewidth=2, label='BCE')
    axes[1, 2].plot(y_pred_prob, focal, 'r-', linewidth=2, label=f'Focal (γ={gamma})')
    axes[1, 2].set_title('Focal Loss (для незбалансованих даних)')
    axes[1, 2].set_xlabel('Predicted Probability')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle('Функції втрат для різних задач', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


visualize_loss_functions()

# ============================================================================
# ЧАСТИНА 3: ОПТИМІЗАТОРИ - ПОРІВНЯННЯ
# ============================================================================

print("\n" + "=" * 60)
print("ЧАСТИНА 3: ПОРІВНЯННЯ ОПТИМІЗАТОРІВ")
print("=" * 60)


def create_loss_landscape():
    """Створює складний ландшафт втрат для візуалізації"""
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x, y)

    # Rosenbrock function - класична тестова функція
    Z = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2

    # Додаємо локальні мінімуми
    Z += 5 * np.sin(2 * X) * np.sin(2 * Y)

    return X, Y, Z


def optimize_2d_function(optimizer_class, optimizer_params, start_point, num_steps=100):
    """Оптимізує 2D функцію за допомогою заданого оптимізатора"""

    # Початкова точка
    params = [torch.tensor([start_point[0]], dtype=torch.float32, requires_grad=True),
              torch.tensor([start_point[1]], dtype=torch.float32, requires_grad=True)]

    # Створюємо оптимізатор
    optimizer = optimizer_class(params, **optimizer_params)

    trajectory = []
    losses = []

    for step in range(num_steps):
        # Зберігаємо позицію
        trajectory.append([params[0].item(), params[1].item()])

        # Обчислюємо втрати (Rosenbrock + sin)
        loss = (1 - params[0]) ** 2 + 100 * (params[1] - params[0] ** 2) ** 2
        loss += 5 * torch.sin(2 * params[0]) * torch.sin(2 * params[1])
        losses.append(loss.item())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Оновлення параметрів
        optimizer.step()

    return np.array(trajectory), losses


def compare_optimizers():
    """Візуалізація траєкторій різних оптимізаторів"""

    # Створюємо ландшафт
    X, Y, Z = create_loss_landscape()

    # Початкова точка
    start_point = [-2.0, 2.0]

    # Налаштування оптимізаторів
    optimizers = {
        'SGD': (optim.SGD, {'lr': 0.001}),
        'SGD + Momentum': (optim.SGD, {'lr': 0.001, 'momentum': 0.9}),
        'RMSprop': (optim.RMSprop, {'lr': 0.01}),
        'Adam': (optim.Adam, {'lr': 0.1}),
        'AdaGrad': (optim.Adagrad, {'lr': 0.5}),
    }

    # Візуалізація
    fig = plt.figure(figsize=(18, 12))

    for idx, (name, (opt_class, opt_params)) in enumerate(optimizers.items()):
        # Траєкторія
        ax = plt.subplot(2, 3, idx + 1)

        # Контурний графік
        contour = ax.contour(X, Y, Z, levels=30, alpha=0.4, cmap='viridis')
        ax.clabel(contour, inline=True, fontsize=8)

        # Оптимізація
        trajectory, losses = optimize_2d_function(opt_class, opt_params, start_point, num_steps=50)

        # Візуалізація траєкторії
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'r.-', linewidth=2, markersize=4)
        ax.plot(start_point[0], start_point[1], 'go', markersize=10, label='Start')
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'r*', markersize=15, label='End')

        ax.set_title(f'{name}\nFinal loss: {losses[-1]:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Loss curves
    ax = plt.subplot(2, 3, 6)

    for name, (opt_class, opt_params) in optimizers.items():
        _, losses = optimize_2d_function(opt_class, opt_params, start_point, num_steps=50)
        ax.plot(losses, label=name, linewidth=2)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Convergence Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.suptitle('Порівняння оптимізаторів на Rosenbrock Function', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


compare_optimizers()

# ============================================================================
# ЧАСТИНА 4: LEARNING RATE SCHEDULING
# ============================================================================

print("\n" + "=" * 60)
print("ЧАСТИНА 4: LEARNING RATE SCHEDULING")
print("=" * 60)


def visualize_lr_schedules():
    """Візуалізація різних стратегій зміни learning rate"""

    epochs = 100
    initial_lr = 0.1

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Constant
    lr_constant = [initial_lr] * epochs
    axes[0, 0].plot(lr_constant, 'b-', linewidth=2)
    axes[0, 0].set_title('Constant LR')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Learning Rate')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Step Decay
    lr_step = []
    lr = initial_lr
    for epoch in range(epochs):
        if epoch % 30 == 0 and epoch > 0:
            lr *= 0.5
        lr_step.append(lr)

    axes[0, 1].plot(lr_step, 'b-', linewidth=2)
    axes[0, 1].set_title('Step Decay (every 30 epochs)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Learning Rate')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Exponential Decay
    lr_exp = [initial_lr * (0.95 ** epoch) for epoch in range(epochs)]
    axes[0, 2].plot(lr_exp, 'b-', linewidth=2)
    axes[0, 2].set_title('Exponential Decay')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Learning Rate')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Cosine Annealing
    lr_cosine = [initial_lr * (1 + np.cos(np.pi * epoch / epochs)) / 2
                 for epoch in range(epochs)]
    axes[1, 0].plot(lr_cosine, 'b-', linewidth=2)
    axes[1, 0].set_title('Cosine Annealing')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Warm-up + Decay
    lr_warmup = []
    warmup_epochs = 10
    for epoch in range(epochs):
        if epoch < warmup_epochs:
            lr = initial_lr * (epoch + 1) / warmup_epochs
        else:
            lr = initial_lr * (0.95 ** (epoch - warmup_epochs))
        lr_warmup.append(lr)

    axes[1, 1].plot(lr_warmup, 'b-', linewidth=2)
    axes[1, 1].axvline(x=warmup_epochs, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Warm-up + Exponential Decay')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Cyclic (Triangle)
    lr_cyclic = []
    cycle_length = 20
    for epoch in range(epochs):
        cycle = epoch % cycle_length
        if cycle < cycle_length / 2:
            lr = initial_lr * 0.1 + (initial_lr - initial_lr * 0.1) * cycle / (cycle_length / 2)
        else:
            lr = initial_lr - (initial_lr - initial_lr * 0.1) * (cycle - cycle_length / 2) / (cycle_length / 2)
        lr_cyclic.append(lr)

    axes[1, 2].plot(lr_cyclic, 'b-', linewidth=2)
    axes[1, 2].set_title('Cyclic LR')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Learning Rate')
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle('Стратегії зміни Learning Rate', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


visualize_lr_schedules()


# Learning Rate Finder
def find_optimal_lr(model, dataloader, start_lr=1e-7, end_lr=10, num_iter=100):
    """
    Реалізація Learning Rate Finder
    """
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=start_lr)

    lrs = []
    losses = []

    # Експоненційне збільшення lr
    lr_mult = (end_lr / start_lr) ** (1 / num_iter)

    current_lr = start_lr
    best_loss = float('inf')

    for batch_idx, (data, target) in enumerate(dataloader):
        if batch_idx >= num_iter:
            break

        # Встановлюємо lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)

        # Якщо втрати вибухають - зупиняємо
        if loss.item() > best_loss * 4:
            break

        if loss.item() < best_loss:
            best_loss = loss.item()

        losses.append(loss.item())
        lrs.append(current_lr)

        loss.backward()
        optimizer.step()

        current_lr *= lr_mult

    # Візуалізація
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.grid(True, alpha=0.3)

    # Знаходимо точку з найбільшим спадом
    gradients = np.gradient(losses)
    best_lr_idx = np.argmin(gradients)
    best_lr = lrs[best_lr_idx]

    plt.axvline(x=best_lr, color='r', linestyle='--', label=f'Suggested LR: {best_lr:.2e}')
    plt.legend()
    plt.show()

    return best_lr


print("\n💡 Learning Rate Finder допомагає знайти оптимальний початковий learning rate")
print("Шукаємо точку, де loss починає швидко падати, але до того як починає розходитися")

# ============================================================================
# ЧАСТИНА 5: ДІАГНОСТИКА ПРОБЛЕМ НАВЧАННЯ
# ============================================================================

print("\n" + "=" * 60)
print("ЧАСТИНА 5: ДІАГНОСТИКА ПРОБЛЕМ")
print("=" * 60)


class TrainingDiagnostics:
    """Клас для діагностики типових проблем навчання"""

    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'gradients': [],
            'weights': []
        }

    def diagnose_vanishing_gradient(self, gradients):
        """Детектує vanishing gradient problem"""
        grad_norms = [torch.norm(g).item() for g in gradients]
        avg_norm = np.mean(grad_norms)

        if avg_norm < 1e-5:
            return "⚠️ VANISHING GRADIENT: Градієнти занадто малі!"
        elif avg_norm < 1e-3:
            return "⚡ Попередження: Градієнти малі, можливі проблеми"
        else:
            return "✅ Градієнти в нормі"

    def diagnose_exploding_gradient(self, gradients):
        """Детектує exploding gradient problem"""
        grad_norms = [torch.norm(g).item() for g in gradients]
        max_norm = np.max(grad_norms)

        if max_norm > 100:
            return "🔥 EXPLODING GRADIENT: Градієнти вибухають!"
        elif max_norm > 10:
            return "⚡ Попередження: Великі градієнти"
        else:
            return "✅ Градієнти стабільні"

    def diagnose_overfitting(self, train_loss, val_loss, threshold=0.1):
        """Детектує overfitting"""
        if len(train_loss) < 10 or len(val_loss) < 10:
            return "Недостатньо даних для діагностики"

        recent_train = np.mean(train_loss[-10:])
        recent_val = np.mean(val_loss[-10:])

        if recent_val > recent_train * (1 + threshold):
            gap = (recent_val - recent_train) / recent_train * 100
            return f"📈 OVERFITTING: Val loss на {gap:.1f}% вище train loss!"
        else:
            return "✅ Немає ознак overfitting"

    def diagnose_underfitting(self, train_loss, target_loss):
        """Детектує underfitting"""
        if len(train_loss) < 10:
            return "Недостатньо даних для діагностики"

        recent_train = np.mean(train_loss[-10:])

        if recent_train > target_loss * 1.5:
            return f"📉 UNDERFITTING: Train loss все ще високий ({recent_train:.4f})!"
        else:
            return "✅ Модель навчається нормально"

    def visualize_diagnostics(self, model_name="Model"):
        """Візуалізація діагностики навчання"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Loss curves
        axes[0, 0].plot(self.history['train_loss'], 'b-', label='Train')
        axes[0, 0].plot(self.history['val_loss'], 'r-', label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training vs Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Gradient norms
        if self.history['gradients']:
            grad_norms = [np.mean([torch.norm(g).item() for g in grads])
                          for grads in self.history['gradients']]
            axes[0, 1].plot(grad_norms, 'g-')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Gradient Norm')
            axes[0, 1].set_title('Gradient Norm Evolution')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Weight distribution
        if self.history['weights']:
            latest_weights = self.history['weights'][-1]
            all_weights = torch.cat([w.flatten() for w in latest_weights])
            axes[1, 0].hist(all_weights.detach().cpu().numpy(), bins=50, alpha=0.7, color='purple')
            axes[1, 0].set_xlabel('Weight Value')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Weight Distribution')
            axes[1, 0].grid(True, alpha=0.3)

        # 4. Learning dynamics
        if len(self.history['train_loss']) > 1:
            loss_changes = np.diff(self.history['train_loss'])
            axes[1, 1].plot(loss_changes, 'orange')
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss Change')
            axes[1, 1].set_title('Loss Change per Epoch')
            axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f'Training Diagnostics: {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()


# Демонстрація діагностики
def demonstrate_training_problems():
    """Симуляція та діагностика різних проблем навчання"""

    print("\n🔍 ДЕМОНСТРАЦІЯ ТИПОВИХ ПРОБЛЕМ НАВЧАННЯ\n")

    diagnostics = TrainingDiagnostics()

    # 1. Vanishing Gradient
    print("1. Vanishing Gradient Problem:")
    small_gradients = [torch.randn(10, 10) * 1e-6 for _ in range(5)]
    print(f"   {diagnostics.diagnose_vanishing_gradient(small_gradients)}")

    # 2. Exploding Gradient
    print("\n2. Exploding Gradient Problem:")
    large_gradients = [torch.randn(10, 10) * 1000 for _ in range(5)]
    print(f"   {diagnostics.diagnose_exploding_gradient(large_gradients)}")

    # 3. Overfitting
    print("\n3. Overfitting Detection:")
    train_loss = [0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04, 0.03, 0.02]
    val_loss = [0.5, 0.35, 0.3, 0.28, 0.27, 0.28, 0.30, 0.35, 0.40, 0.45]
    print(f"   {diagnostics.diagnose_overfitting(train_loss, val_loss)}")

    # 4. Underfitting
    print("\n4. Underfitting Detection:")
    high_train_loss = [0.8, 0.75, 0.72, 0.70, 0.68, 0.67, 0.66, 0.65, 0.64, 0.63]
    print(f"   {diagnostics.diagnose_underfitting(high_train_loss, target_loss=0.1)}")


demonstrate_training_problems()

# ============================================================================
# ЧАСТИНА 6: GRADIENT CLIPPING
# ============================================================================

print("\n" + "=" * 60)
print("ЧАСТИНА 6: GRADIENT CLIPPING")
print("=" * 60)


def demonstrate_gradient_clipping():
    """Демонстрація gradient clipping"""

    # Створюємо модель з великими градієнтами
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )

    # Симулюємо великі градієнти
    for param in model.parameters():
        param.grad = torch.randn_like(param) * 100  # Великі градієнти!

    print("📊 Градієнти ДО clipping:")
    total_norm_before = 0
    for param in model.parameters():
        param_norm = param.grad.norm(2)
        total_norm_before += param_norm.item() ** 2
    total_norm_before = total_norm_before ** 0.5
    print(f"   Total gradient norm: {total_norm_before:.2f}")

    # Gradient clipping
    max_norm = 1.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    print(f"\n✂️ Застосовуємо gradient clipping (max_norm={max_norm})")

    print("\n📊 Градієнти ПІСЛЯ clipping:")
    total_norm_after = 0
    for param in model.parameters():
        param_norm = param.grad.norm(2)
        total_norm_after += param_norm.item() ** 2
    total_norm_after = total_norm_after ** 0.5
    print(f"   Total gradient norm: {total_norm_after:.2f}")

    print(f"\n✅ Зменшення норми: {total_norm_before / total_norm_after:.2f}x")


demonstrate_gradient_clipping()

# ============================================================================
# ЧАСТИНА 7: ПОВНИЙ ТРЕНУВАЛЬНИЙ PIPELINE
# ============================================================================

print("\n" + "=" * 60)
print("ЧАСТИНА 7: ПОВНИЙ ТРЕНУВАЛЬНИЙ PIPELINE")
print("=" * 60)


class CompletePipeline:
    """Повний pipeline навчання з усіма best practices"""

    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.diagnostics = TrainingDiagnostics()

    def train_epoch(self, dataloader, optimizer, criterion, clip_grad=None):
        """Одна епоха навчання"""
        self.model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)

            # Backward pass
            loss.backward()

            # Gradient clipping якщо потрібно
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)

            # Зберігаємо градієнти для діагностики
            if batch_idx == 0:  # Тільки перший батч кожної епохи
                grads = [param.grad.clone() for param in self.model.parameters() if param.grad is not None]
                self.diagnostics.history['gradients'].append(grads)

            # Update
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, dataloader, criterion):
        """Валідація моделі"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(self, train_loader, val_loader, epochs=10, lr=0.001,
              optimizer_name='Adam', scheduler_type='StepLR'):
        """Повний цикл навчання"""

        # Вибір оптимізатора
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        else:
            optimizer = optim.RMSprop(self.model.parameters(), lr=lr)

        # Learning rate scheduler
        if scheduler_type == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif scheduler_type == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        else:
            scheduler = None

        criterion = nn.CrossEntropyLoss()

        print(f"🚀 Навчання з {optimizer_name} оптимізатором та {scheduler_type} scheduler")
        print("=" * 50)

        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader, optimizer, criterion, clip_grad=1.0)
            self.diagnostics.history['train_loss'].append(train_loss)

            # Validation
            val_loss = self.validate(val_loader, criterion)
            self.diagnostics.history['val_loss'].append(val_loss)

            # Learning rate scheduling
            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = lr

            # Зберігаємо ваги для діагностики
            weights = [param.clone() for param in self.model.parameters()]
            self.diagnostics.history['weights'].append(weights)

            # Виведення прогресу
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"LR: {current_lr:.6f}")

                # Діагностика
                if len(self.diagnostics.history['gradients']) > 0:
                    grad_check = self.diagnostics.diagnose_exploding_gradient(
                        self.diagnostics.history['gradients'][-1]
                    )
                    if "EXPLODING" in grad_check or "Попередження" in grad_check:
                        print(f"   {grad_check}")

        print("\n📊 Фінальна діагностика:")
        print(f"   {self.diagnostics.diagnose_overfitting(self.diagnostics.history['train_loss'],
                                                          self.diagnostics.history['val_loss'])}")

        # Візуалізація
        self.diagnostics.visualize_diagnostics(f"{optimizer_name} + {scheduler_type}")

        return self.diagnostics.history


# Тест pipeline на простій задачі
def _complete_pipeline_test():
    """Тестування повного pipeline"""

    # Генеруємо дані
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                               n_redundant=5, n_classes=3, random_state=42)

    X_train, X_val = X[:800], X[800:]
    y_train, y_val = y[:800], y[800:]

    # Створюємо DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Створюємо модель
    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, 3)
    )

    # Навчання
    pipeline = CompletePipeline(model, device=device)
    history = pipeline.train(train_loader, val_loader, epochs=30,
                             lr=0.01, optimizer_name='Adam',
                             scheduler_type='CosineAnnealingLR')


print("\n🧪 Тестуємо повний pipeline на синтетичних даних:\n")
_complete_pipeline_test()

# ============================================================================
# ПІДСУМКИ
# ============================================================================

print("\n" + "=" * 60)
print("ПІДСУМКИ")
print("=" * 60)

print("""
✅ Що ми вивчили:

1. **Backpropagation**
   - Покрокове обчислення градієнтів
   - Chain rule в дії
   - Збереження проміжних значень

2. **Функції втрат**
   - MSE, MAE, Huber для регресії
   - BCE, Cross-Entropy для класифікації
   - Focal Loss для незбалансованих даних

3. **Оптимізатори**
   - SGD, Momentum, RMSprop, Adam
   - Візуалізація траєкторій оптимізації
   - Вибір оптимізатора для задачі

4. **Learning Rate**
   - Різні стратегії scheduling
   - Learning Rate Finder
   - Warm-up стратегії

5. **Діагностика проблем**
   - Vanishing/Exploding gradients
   - Overfitting/Underfitting
   - Gradient clipping

6. **Best Practices**
   - Повний тренувальний pipeline
   - Моніторинг метрик
   - Візуалізація процесу навчання

🎯 Ключові висновки:
- Adam - хороший вибір за замовчуванням
- Learning rate - найважливіший гіперпараметр
- Gradient clipping важливий для RNN
- Завжди слідкуйте за train vs val loss
- Використовуйте scheduler для кращої збіжності

🚀 Наступні кроки:
- Регуляризація (Dropout, L2, BatchNorm)
- Data Augmentation
- Transfer Learning
- Hyperparameter Tuning
""")