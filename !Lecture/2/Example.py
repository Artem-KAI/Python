#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Практичний ноутбук до Лекції 2: Персептрон та багатошарові мережі
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
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import seaborn as sns

# Для MNIST
from torchvision import datasets, transforms

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
# ЧАСТИНА 1: ФУНКЦІЇ АКТИВАЦІЇ
# ============================================================================

print("=" * 60)
print("ЧАСТИНА 1: ФУНКЦІЇ АКТИВАЦІЇ")
print("=" * 60)


def plot_activation_functions():
    """Візуалізація основних функцій активації"""

    x = np.linspace(-3, 3, 100)

    # Визначення функцій
    def step(x):
        return (x >= 0).astype(float)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def tanh(x):
        return np.tanh(x)

    def relu(x):
        return np.maximum(0, x)

    def leaky_relu(x, alpha=0.1):
        return np.where(x > 0, x, alpha * x)

    # Візуалізація
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    functions = [
        (step, "Step Function", "binary"),
        (sigmoid, "Sigmoid", "smooth"),
        (tanh, "Tanh", "smooth"),
        (relu, "ReLU", "linear"),
        (lambda x: leaky_relu(x), "Leaky ReLU", "linear"),
        (lambda x: x * sigmoid(x), "Swish", "smooth")
    ]

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    for ax, (func, name, type_), color in zip(axes, functions, colors):
        y = func(x)
        ax.plot(x, y, color=color, linewidth=2.5, label=name)
        ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='black', linewidth=0.5, alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_xlabel('z')
        ax.set_ylabel('f(z)')
        ax.set_ylim([-1.5, 2])

        # Додаємо характеристики
        if name == "Sigmoid":
            ax.text(0, -1.3, "Range: (0, 1)", ha='center', fontsize=10)
        elif name == "Tanh":
            ax.text(0, -1.3, "Range: (-1, 1)", ha='center', fontsize=10)
        elif name == "ReLU":
            ax.text(0, -1.3, "Range: [0, ∞)", ha='center', fontsize=10)

    plt.suptitle("Функції активації в нейронних мережах", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


plot_activation_functions()





# ============================================================================
# ЧАСТИНА 2: ПЕРСЕПТРОН З НУЛЯ
# ============================================================================

print("\n" + "=" * 60)
print("ЧАСТИНА 2: РЕАЛІЗАЦІЯ ПЕРСЕПТРОНА")
print("=" * 60)


class PerceptronNumPy:
    """Персептрон реалізований на NumPy"""

    def __init__(self, input_size, learning_rate=0.01):
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0.0
        self.lr = learning_rate
        self.history = {'loss': [], 'accuracy': []}

    def activation(self, z):
        """Крокова функція активації"""
        return (z >= 0).astype(int)

    def predict(self, X):
        """Передбачення для батчу прикладів"""
        z = X @ self.weights + self.bias
        return self.activation(z)

    def train_step(self, X, y):
        """Один крок навчання"""
        predictions = self.predict(X)
        errors = y - predictions

        # Perceptron learning rule
        self.weights += self.lr * (X.T @ errors) / len(X)
        self.bias += self.lr * np.mean(errors)

        return errors

    def fit(self, X, y, epochs=100, verbose=True):
        """Повний цикл навчання"""
        for epoch in range(epochs):
            errors = self.train_step(X, y)

            loss = np.mean(errors ** 2)
            accuracy = np.mean(self.predict(X) == y)

            self.history['loss'].append(loss)
            self.history['accuracy'].append(accuracy)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Accuracy: {accuracy:.2%}")

        return self.history


# Тест на простих логічних функціях
def test_logical_functions():
    """Тестування персептрона на AND, OR, XOR"""

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Цільові значення для різних функцій
    targets = {
        'AND': np.array([0, 0, 0, 1]),
        'OR': np.array([0, 1, 1, 1]),
        'XOR': np.array([0, 1, 1, 0])
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (name, y) in zip(axes, targets.items()):
        perceptron = PerceptronNumPy(2, learning_rate=0.1)
        history = perceptron.fit(X, y, epochs=100, verbose=False)

        # Візуалізація результатів
        predictions = perceptron.predict(X)

        # Scatter plot
        colors = ['red' if yi == 0 else 'blue' for yi in y]
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors='black', linewidth=2)

        # Decision boundary (якщо можливо)
        if perceptron.weights[1] != 0:
            x_boundary = np.linspace(-0.5, 1.5, 100)
            y_boundary = -(perceptron.weights[0] * x_boundary + perceptron.bias) / perceptron.weights[1]
            ax.plot(x_boundary, y_boundary, 'g--', linewidth=2, alpha=0.7, label='Decision boundary')

        # Додаємо передбачення
        for i, (xi, yi, pred) in enumerate(zip(X, y, predictions)):
            ax.annotate(f'True: {yi}\nPred: {pred}',
                        xy=(xi[0], xi[1]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=9)

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_title(f'{name} Function\nAccuracy: {history["accuracy"][-1]:.0%}')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle("Персептрон на логічних функціях", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


test_logical_functions()

# ============================================================================
# ЧАСТИНА 3: БАГАТОШАРОВА МЕРЕЖА (MLP) З НУЛЯ
# ============================================================================

print("\n" + "=" * 60)
print("ЧАСТИНА 3: MLP ДЛЯ ВИРІШЕННЯ XOR")
print("=" * 60)


class MLPNumPy:
    """Двошарова мережа для XOR"""

    def __init__(self, input_size=2, hidden_size=4, output_size=1, lr=0.1):
        # Ініціалізація ваг (Xavier initialization)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        self.lr = lr
        self.history = {'loss': []}

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def forward(self, X):
        """Forward pass"""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y):
        """Backward pass (backpropagation)"""
        m = X.shape[0]

        # Градієнт вихідного шару
        dz2 = self.a2 - y.reshape(-1, 1)
        dW2 = (1 / m) * self.a1.T @ dz2
        db2 = (1 / m) * np.sum(dz2, axis=0, keepdims=True)

        # Градієнт прихованого шару
        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = (1 / m) * X.T @ dz1
        db1 = (1 / m) * np.sum(dz1, axis=0, keepdims=True)

        # Оновлення ваг
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=1000, verbose=True):
        """Навчання мережі"""
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)

            # Compute loss (MSE)
            loss = np.mean((output - y.reshape(-1, 1)) ** 2)
            self.history['loss'].append(loss)

            # Backward pass
            self.backward(X, y)

            if verbose and epoch % 100 == 0:
                accuracy = np.mean((output > 0.5).astype(int).flatten() == y)
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {accuracy:.2%}")

        return self.history


# Вирішення XOR з MLP
def solve_xor_with_mlp():
    """Демонстрація вирішення XOR за допомогою MLP"""

    # Дані XOR
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])  # XOR

    # Навчання MLP
    print("Навчання MLP на XOR:")
    mlp = MLPNumPy(input_size=2, hidden_size=4, output_size=1, lr=0.5)
    history = mlp.train(X, y, epochs=1000)

    # Результати
    print("\nФінальні передбачення:")
    predictions = mlp.forward(X)
    for xi, yi, pred in zip(X, y, predictions):
        print(f"Input: {xi} | True: {yi} | Predicted: {pred[0]:.3f} | Rounded: {int(pred[0] > 0.5)}")

    # Візуалізація
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Loss curve
    axes[0].plot(history['loss'], 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)

    # 2. Decision boundary
    h = 0.01
    xx, yy = np.meshgrid(np.arange(-0.5, 1.5, h), np.arange(-0.5, 1.5, h))
    Z = mlp.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axes[1].contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6)
    axes[1].contour(xx, yy, Z, levels=[0.5], colors='green', linewidths=2)

    # Точки даних
    colors = ['red' if yi == 0 else 'blue' for yi in y]
    axes[1].scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors='black', linewidth=2)
    axes[1].set_xlabel('x₁')
    axes[1].set_ylabel('x₂')
    axes[1].set_title('Decision Boundary for XOR')

    # 3. Network architecture visualization
    axes[2].axis('off')
    axes[2].set_title('Network Architecture')

    # Візуалізація архітектури
    layers = [2, 4, 1]
    layer_positions = [0.2, 0.5, 0.8]

    for layer_idx, (n_neurons, x_pos) in enumerate(zip(layers, layer_positions)):
        y_positions = np.linspace(0.2, 0.8, n_neurons)

        for neuron_idx, y_pos in enumerate(y_positions):
            circle = plt.Circle((x_pos, y_pos), 0.03, color='lightblue', ec='black', linewidth=2)
            axes[2].add_patch(circle)

            # Connections
            if layer_idx < len(layers) - 1:
                next_layer_size = layers[layer_idx + 1]
                next_y_positions = np.linspace(0.2, 0.8, next_layer_size)
                next_x_pos = layer_positions[layer_idx + 1]

                for next_y in next_y_positions:
                    axes[2].plot([x_pos, next_x_pos], [y_pos, next_y],
                                 'gray', alpha=0.3, linewidth=0.5)

    # Labels
    axes[2].text(0.2, 0.1, 'Input\n(2)', ha='center', fontsize=10, fontweight='bold')
    axes[2].text(0.5, 0.1, 'Hidden\n(4)', ha='center', fontsize=10, fontweight='bold')
    axes[2].text(0.8, 0.1, 'Output\n(1)', ha='center', fontsize=10, fontweight='bold')

    plt.suptitle("MLP вирішує XOR", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


solve_xor_with_mlp()

# ============================================================================
# ЧАСТИНА 4: MLP В PYTORCH
# ============================================================================

print("\n" + "=" * 60)
print("ЧАСТИНА 4: MLP В PYTORCH")
print("=" * 60)


class MLPPyTorch(nn.Module):
    """MLP реалізована в PyTorch"""

    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
        super(MLPPyTorch, self).__init__()

        layers = []
        prev_size = input_size

        # Створюємо приховані шари
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))

            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())

            prev_size = hidden_size

        # Вихідний шар
        layers.append(nn.Linear(prev_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Порівняння різних архітектур
def compare_architectures():
    """Порівнюємо різні архітектури MLP"""

    # Генеруємо складніші дані (спіраль)
    np.random.seed(42)
    n_points = 500

    # Клас 0 - внутрішня спіраль
    theta0 = np.linspace(0, 4 * np.pi, n_points // 2) + np.random.randn(n_points // 2) * 0.1
    r0 = theta0 / (4 * np.pi) * 2
    x0 = r0 * np.cos(theta0)
    y0 = r0 * np.sin(theta0)

    # Клас 1 - зовнішня спіраль
    theta1 = np.linspace(0, 4 * np.pi, n_points // 2) + np.random.randn(n_points // 2) * 0.1
    r1 = theta1 / (4 * np.pi) * 2 + 0.5
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)

    # Об'єднуємо дані
    X = np.vstack([np.column_stack([x0, y0]), np.column_stack([x1, y1])])
    y = np.hstack([np.zeros(n_points // 2), np.ones(n_points // 2)])

    # Перемішуємо
    indices = np.random.permutation(n_points)
    X = X[indices]
    y = y[indices]

    # Конвертуємо в тензори
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1)

    # Різні архітектури
    architectures = {
        'Shallow (1x16)': [16],
        'Deep (4x8)': [8, 8, 8, 8],
        'Wide (1x64)': [64],
        'Pyramid (32-16-8)': [32, 16, 8]
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for idx, (name, hidden_sizes) in enumerate(architectures.items()):
        print(f"\nНавчання {name}...")

        model = MLPPyTorch(2, hidden_sizes, 1, activation='relu')
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()

        losses = []

        # Навчання
        for epoch in range(500):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Візуалізація результатів
        ax_data = axes[0, idx]
        ax_loss = axes[1, idx]

        # Decision boundary
        h = 0.02
        xx, yy = np.meshgrid(np.arange(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, h),
                             np.arange(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, h))

        with torch.no_grad():
            Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))
            Z = torch.sigmoid(Z).numpy().reshape(xx.shape)

        ax_data.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6)
        ax_data.contour(xx, yy, Z, levels=[0.5], colors='green', linewidths=2)
        ax_data.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='black', linewidth=0.5)
        ax_data.set_title(f'{name}')
        ax_data.set_xlabel('x₁')
        ax_data.set_ylabel('x₂')

        # Loss curve
        ax_loss.plot(losses, 'b-', linewidth=1)
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('BCE Loss')
        ax_loss.set_title(f'Final Loss: {losses[-1]:.4f}')
        ax_loss.grid(True, alpha=0.3)

    plt.suptitle("Порівняння різних архітектур MLP на спіральних даних", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


compare_architectures()

# ============================================================================
# ЧАСТИНА 5: MNIST КЛАСИФІКАЦІЯ
# ============================================================================

print("\n" + "=" * 60)
print("ЧАСТИНА 5: КЛАСИФІКАЦІЯ MNIST")
print("=" * 60)

# Завантаження MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Mean and std of MNIST
])

# Завантаження датасету
print("Завантаження MNIST...")
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f"Тренувальний набір: {len(train_dataset)} зображень")
print(f"Тестовий набір: {len(test_dataset)} зображень")


# Візуалізація прикладів
def visualize_mnist_samples():
    """Показуємо приклади з MNIST"""
    fig, axes = plt.subplots(2, 10, figsize=(15, 4))

    for i in range(20):
        img, label = train_dataset[i]
        ax = axes[i // 10, i % 10]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(f'{label}')
        ax.axis('off')

    plt.suptitle("Приклади з MNIST", fontsize=14)
    plt.tight_layout()
    plt.show()


visualize_mnist_samples()


# Модель для MNIST
class MNISTNet(nn.Module):
    """MLP для класифікації MNIST"""

    def __init__(self, hidden_sizes=[128, 64]):
        super(MNISTNet, self).__init__()

        self.flatten = nn.Flatten()

        layers = []
        input_size = 784  # 28x28

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # Regularization
            input_size = hidden_size

        layers.append(nn.Linear(input_size, 10))  # 10 classes

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        return self.classifier(x)


# Функції навчання та тестування
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Одна епоха навчання"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    return total_loss / len(dataloader), 100. * correct / total


def test_model(model, dataloader, criterion, device):
    """Тестування моделі"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    return test_loss / len(dataloader), 100. * correct / total


# Навчання моделі
def train_mnist_model():
    """Повний pipeline навчання MNIST"""

    model = MNISTNet(hidden_sizes=[256, 128, 64]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"\nМодель: {sum(p.numel() for p in model.parameters())} параметрів")
    print("Починаємо навчання...")

    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    epochs = 10

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = test_model(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f'Epoch {epoch + 1:2d}/{epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
              f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')

    # Візуалізація результатів
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(train_losses, 'b-', label='Train Loss')
    axes[0].plot(test_losses, 'r-', label='Test Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Test Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_accs, 'b-', label='Train Accuracy')
    axes[1].plot(test_accs, 'r-', label='Test Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Test Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'MNIST Training Results - Final Test Accuracy: {test_accs[-1]:.2f}%',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return model


trained_model = train_mnist_model()


# Візуалізація помилок
def visualize_errors(model, dataloader, device, n_errors=10):
    """Показуємо приклади помилкових класифікацій"""

    model.eval()
    errors = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)

            # Знаходимо помилки
            incorrect = predicted != target

            for idx in torch.where(incorrect)[0]:
                if len(errors) < n_errors:
                    errors.append({
                        'image': data[idx].cpu(),
                        'true': target[idx].item(),
                        'pred': predicted[idx].item(),
                        'probs': torch.softmax(output[idx], dim=0).cpu()
                    })
                else:
                    break

            if len(errors) >= n_errors:
                break

    # Візуалізація
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for idx, error in enumerate(errors):
        ax = axes[idx]
        ax.imshow(error['image'].squeeze(), cmap='gray')
        ax.set_title(f"True: {error['true']}, Pred: {error['pred']}\n"
                     f"Conf: {error['probs'][error['pred']]:.2%}")
        ax.axis('off')

    plt.suptitle("Приклади помилкових класифікацій", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


visualize_errors(trained_model, test_loader, device)

# ============================================================================
# ПІДСУМКИ
# ============================================================================

print("\n" + "=" * 60)
print("ПІДСУМКИ")
print("=" * 60)

print("""
✅ Що ми вивчили:
1. Функції активації та їх властивості
2. Реалізація персептрона з нуля
3. Обмеження персептрона (XOR проблема)
4. Багатошарові мережі (MLP)
5. Forward pass та backpropagation
6. Класифікація MNIST з високою точністю

📊 Результати:
- Персептрон: вирішує AND, OR але не XOR
- MLP: успішно вирішує XOR
- MNIST: ~98% точності з простою MLP

🎯 Наступні кроки:
- Детальне вивчення backpropagation
- Convolutional Neural Networks (CNN)
- Регуляризація та оптимізація
- Transfer learning
""")
      
"""Візуалізація градієнтів функцій активації"""

x = np.linspace(-3, 3, 100)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Sigmoid та її похідна
sigmoid = 1 / (1 + np.exp(-x))
sigmoid_grad = sigmoid * (1 - sigmoid)

axes[0].plot(x, sigmoid, 'b-', label='Sigmoid', linewidth=2)
axes[0].plot(x, sigmoid_grad, 'r--', label='Gradient', linewidth=2)
axes[0].set_title('Sigmoid: Vanishing Gradient Problem')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Tanh та її похідна
tanh = np.tanh(x)
tanh_grad = 1 - tanh ** 2

axes[1].plot(x, tanh, 'g-', label='Tanh', linewidth=2)
axes[1].plot(x, tanh_grad, 'r--', label='Gradient', linewidth=2)
axes[1].set_title('Tanh: Better than Sigmoid')
axes[1].legend()
axes[1].grid(True, alpha=0.3)