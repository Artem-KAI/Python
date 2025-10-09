#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Практичний ноутбук до Лекції 1: Вступ до глибокого навчання
Курс: Deep Learning
Автор: Зівакін Валерій Дмитрович

Цей ноутбук в майбутньому має запускатися в (не забудьте встановити залежності!):
- Google Colab (рекомендовано)
- Локально з GPU
- Kaggle Notebooks
"""

# ============================================================================
# ЧАСТИНА 0: НАЛАШТУВАННЯ СЕРЕДОВИЩА
# ============================================================================

# Перевірка, чи ми в Google Colab
import sys

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("🎉 Запущено в Google Colab!")
    print("GPU вже має бути активована (Runtime -> Change runtime type -> GPU)")
else:
    print("💻 Запущено локально")

# Основні імпорти
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Налаштування matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12


# ============================================================================
# ЧАСТИНА 1: ПЕРЕВІРКА НАЛАШТУВАННЯ
# ============================================================================

def check_environment():
    """Перевірка середовища виконання"""
    print("=" * 60)
    print("ІНФОРМАЦІЯ ПРО СЕРЕДОВИЩЕ")
    print("=" * 60)

    # PyTorch версія
    print(f"📦 PyTorch версія: {torch.__version__}")

    # CUDA доступність
    cuda_available = torch.cuda.is_available()
    print(f"🖥️  CUDA доступна: {cuda_available}")

    if cuda_available:
        print(f"🔧 CUDA версія: {torch.version.cuda}")
        print(f"📊 Кількість GPU: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Назва: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Пам'ять: {props.total_memory / 1024 ** 3:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
    else:
        print("⚠️  GPU не знайдено. Використовуємо CPU.")
        print("Для активації GPU в Colab: Runtime -> Change runtime type -> GPU")

    # CPU інформація
    print(f"\n💻 CPU threads: {torch.get_num_threads()}")

    return cuda_available


# Запуск перевірки
cuda_available = check_environment()

# Встановлення device
device = torch.device("cuda" if cuda_available else "cpu")
print(f"\n✅ Використовуємо device: {device}")

# ============================================================================
# ЧАСТИНА 2: ОСНОВИ ТЕНЗОРІВ
# ============================================================================

print("\n" + "=" * 60)
print("ЧАСТИНА 2: ОСНОВИ ТЕНЗОРІВ")
print("=" * 60)


# 2.1 Створення тензорів різними способами
def tensor_creation_demo():
    """Демонстрація створення тензорів"""
    print("\n📝 Способи створення тензорів:\n")

    # З Python списку
    tensor_list = torch.tensor([1, 2, 3, 4, 5])
    print(f"З списку: {tensor_list}")

    # З NumPy
    numpy_array = np.array([[1, 2], [3, 4]])
    tensor_numpy = torch.from_numpy(numpy_array)
    print(f"З NumPy: {tensor_numpy}")

    # Спеціальні тензори
    zeros = torch.zeros(2, 3)
    ones = torch.ones(2, 3)
    rand = torch.rand(2, 3)  # [0, 1)
    randn = torch.randn(2, 3)  # N(0, 1)

    print(f"\nНулі:\n{zeros}")
    print(f"\nВипадкові [0,1):\n{rand}")
    print(f"\nНормальний розподіл:\n{randn}")

    # Створення на GPU
    if cuda_available:
        gpu_tensor = torch.randn(2, 3, device='cuda')
        print(f"\nТензор на GPU: {gpu_tensor.device}")


tensor_creation_demo()


# 2.2 Операції з тензорами
def tensor_operations_demo():
    """Демонстрація операцій з тензорами"""
    print("\n📊 Операції з тензорами:\n")

    # Створюємо тензори
    a = torch.tensor([[1., 2.], [3., 4.]])
    b = torch.tensor([[5., 6.], [7., 8.]])

    print(f"Тензор A:\n{a}")
    print(f"Тензор B:\n{b}")

    # Арифметичні операції
    print(f"\nA + B:\n{a + b}")
    print(f"\nA * B (element-wise):\n{a * b}")
    print(f"\nA @ B (matrix mult):\n{a @ b}")

    # Статистичні операції
    c = torch.randn(3, 4)
    print(f"\nВипадковий тензор C:\n{c}")
    print(f"Mean: {c.mean():.4f}")
    print(f"Std: {c.std():.4f}")
    print(f"Max: {c.max():.4f}")
    print(f"Min: {c.min():.4f}")

    # Зміна форми
    d = torch.arange(12)
    print(f"\nВихідний: {d}")
    print(f"Reshape (3,4): {d.reshape(3, 4)}")
    print(f"View (2,6): {d.view(2, 6)}")


tensor_operations_demo()


# 2.3 Broadcasting
def broadcasting_demo():
    """Демонстрація broadcasting"""
    print("\n📡 Broadcasting:\n")

    # Вектор + скаляр
    vec = torch.tensor([1, 2, 3])
    scalar = 10
    print(f"Вектор {vec} + скаляр {scalar} = {vec + scalar}")

    # Матриця + вектор
    matrix = torch.randn(3, 4)
    row_vec = torch.randn(4)
    col_vec = torch.randn(3, 1)

    print(f"\nМатриця shape: {matrix.shape}")
    print(f"Рядок shape: {row_vec.shape}")
    print(f"Стовпець shape: {col_vec.shape}")

    result1 = matrix + row_vec  # Broadcasting по рядках
    result2 = matrix + col_vec  # Broadcasting по стовпцях

    print(f"Матриця + рядок: {result1.shape}")
    print(f"Матриця + стовпець: {result2.shape}")


broadcasting_demo()

# ============================================================================
# ЧАСТИНА 3: CPU vs GPU BENCHMARK
# ============================================================================

print("\n" + "=" * 60)
print("ЧАСТИНА 3: ПОРІВНЯННЯ CPU vs GPU")
print("=" * 60)


def benchmark_device(size, device, num_iterations=100):
    """Benchmark операцій на заданому device"""

    # Створення тензорів на device
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Розігрів
    for _ in range(10):
        c = a @ b
        if device == 'cuda':
            torch.cuda.synchronize()

    # Вимірювання часу
    start_time = time.time()

    for _ in range(num_iterations):
        c = torch.matmul(a, b)
        if device == 'cuda':
            torch.cuda.synchronize()

    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / num_iterations

    return avg_time


def compare_devices():
    """Порівняння продуктивності CPU та GPU"""

    if not cuda_available:
        print("⚠️  GPU недоступна. Пропускаємо benchmark.")
        return

    sizes = [128, 256, 512, 1024, 2048]
    cpu_times = []
    gpu_times = []

    print("\n⏱️  Запуск benchmark (це може зайняти хвилину)...\n")

    for size in sizes:
        print(f"Testing size {size}x{size}...")

        # CPU benchmark
        cpu_time = benchmark_device(size, 'cpu', num_iterations=10)
        cpu_times.append(cpu_time)

        # GPU benchmark
        gpu_time = benchmark_device(size, 'cuda', num_iterations=10)
        gpu_times.append(gpu_time)

        speedup = cpu_time / gpu_time
        print(f"  CPU: {cpu_time:.4f}s, GPU: {gpu_time:.4f}s, Speedup: {speedup:.1f}x")

    # Візуалізація
    plt.figure(figsize=(12, 5))

    # Графік часу
    plt.subplot(1, 2, 1)
    plt.plot(sizes, cpu_times, 'b-o', label='CPU', linewidth=2, markersize=8)
    plt.plot(sizes, gpu_times, 'r-o', label='GPU', linewidth=2, markersize=8)
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.title('Matrix Multiplication Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    # Графік прискорення
    plt.subplot(1, 2, 2)
    speedups = [c / g for c, g in zip(cpu_times, gpu_times)]
    plt.bar(range(len(sizes)), speedups, color='green', alpha=0.7)
    plt.xticks(range(len(sizes)), sizes)
    plt.xlabel('Matrix Size')
    plt.ylabel('Speedup (x times)')
    plt.title('GPU Speedup over CPU')
    plt.grid(True, alpha=0.3)

    # Додаємо значення на стовпчики
    for i, (size, speedup) in enumerate(zip(sizes, speedups)):
        plt.text(i, speedup + 0.5, f'{speedup:.1f}x',
                 ha='center', fontweight='bold')

    plt.tight_layout()
    plt.show()


compare_devices()

# ============================================================================
# ЧАСТИНА 4: ПЕРША НЕЙРОННА МЕРЕЖА
# ============================================================================

print("\n" + "=" * 60)
print("ЧАСТИНА 4: ПЕРША ЛІНІЙНА МОДЕЛЬ")
print("=" * 60)


# Генерація синтетичних даних
def generate_synthetic_data(n_samples=1000, n_features=10, noise_level=0.1):
    """Генерує дані для лінійної регресії: y = Xw + b + noise"""

    torch.manual_seed(42)

    # Справжні параметри
    true_weights = torch.randn(n_features, 1) * 2
    true_bias = torch.randn(1) * 0.5

    # Генерація даних
    X = torch.randn(n_samples, n_features)
    noise = torch.randn(n_samples, 1) * noise_level
    y = X @ true_weights + true_bias + noise

    return X, y, true_weights, true_bias


# Створення датасетів
print("📊 Генерація синтетичних даних...")
X_train, y_train, true_w, true_b = generate_synthetic_data(5000, 20, 0.1)
X_val, y_val, _, _ = generate_synthetic_data(1000, 20, 0.1)

print(f"Тренувальні дані: X={X_train.shape}, y={y_train.shape}")
print(f"Валідаційні дані: X={X_val.shape}, y={y_val.shape}")
print(f"Справжні ваги (перші 5): {true_w[:5].squeeze().tolist()}")
print(f"Справжнє зміщення: {true_b.item():.4f}")

# Створення DataLoader
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# Визначення моделі
class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


# Функція навчання
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Одна епоха навчання"""
    model.train()
    total_loss = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Forward pass
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

    return total_loss / len(dataloader.dataset)


def validate_epoch(model, dataloader, criterion, device):
    """Валідація моделі"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)

            total_loss += loss.item() * batch_x.size(0)

    return total_loss / len(dataloader.dataset)


# Навчання моделі
def train_model(device='cpu', epochs=50):
    """Повний цикл навчання"""
    print(f"\n🚀 Навчання на {device.upper()}...")

    # Ініціалізація
    model = LinearRegression(20).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_losses = []
    val_losses = []

    # Progress bar
    pbar = tqdm(range(epochs), desc='Training')

    start_time = time.time()

    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Оновлення progress bar
        pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}'
        })

    training_time = time.time() - start_time

    print(f"✅ Навчання завершено за {training_time:.2f} секунд")
    print(f"📉 Фінальні втрати - Train: {train_losses[-1]:.4f}, Val: {val_losses[-1]:.4f}")

    return model, train_losses, val_losses, training_time


# Навчання на CPU
cpu_model, cpu_train_losses, cpu_val_losses, cpu_time = train_model('cpu', epochs=30)

# Навчання на GPU (якщо доступно)
if cuda_available:
    gpu_model, gpu_train_losses, gpu_val_losses, gpu_time = train_model('cuda', epochs=30)
    print(f"\n⚡ Прискорення GPU: {cpu_time / gpu_time:.2f}x")
else:
    gpu_train_losses = None
    gpu_val_losses = None

# Візуалізація навчання
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(cpu_train_losses, 'b-', label='CPU Train', linewidth=2)
plt.plot(cpu_val_losses, 'b--', label='CPU Val', linewidth=2)

if gpu_train_losses:
    plt.plot(gpu_train_losses, 'r-', label='GPU Train', linewidth=2)
    plt.plot(gpu_val_losses, 'r--', label='GPU Val', linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Progress')
plt.legend()
plt.grid(True, alpha=0.3)

# Порівняння з справжніми вагами
plt.subplot(1, 2, 2)
cpu_weights = cpu_model.linear.weight.detach().cpu().numpy().squeeze()
true_weights = true_w.numpy().squeeze()

x = np.arange(len(cpu_weights))
width = 0.35

plt.bar(x - width / 2, true_weights, width, label='True Weights', color='green', alpha=0.7)
plt.bar(x + width / 2, cpu_weights, width, label='Learned Weights', color='blue', alpha=0.7)

plt.xlabel('Weight Index')
plt.ylabel('Weight Value')
plt.title('True vs Learned Weights')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# ЧАСТИНА 5: ПРАКТИЧНІ ВПРАВИ
# ============================================================================

print("\n" + "=" * 60)
print("ПРАКТИЧНІ ВПРАВИ")
print("=" * 60)

print("""
 Дивитись 1_tasks.ipynb
""")


# ============================================================================
# ЧАСТИНА 6: ДОДАТКОВІ КОРИСНІ ФУНКЦІЇ
# ============================================================================

def gpu_memory_summary():
    """Виводить інформацію про використання пам'яті GPU"""
    if not cuda_available:
        print("GPU недоступна")
        return

    print("\n" + "=" * 60)
    print("GPU MEMORY SUMMARY")
    print("=" * 60)

    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")

        # Очищення кешу
        torch.cuda.empty_cache()
        print(f"  After cache clear: {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")


gpu_memory_summary()

# ============================================================================
# ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ
# ============================================================================

print("\n" + "=" * 60)
print("ЗБЕРЕЖЕННЯ МОДЕЛІ")
print("=" * 60)

# Збереження моделі
model_path = 'linear_model.pth'
torch.save({
    'model_state_dict': cpu_model.state_dict(),
    'train_losses': cpu_train_losses,
    'val_losses': cpu_val_losses
}, model_path)

print(f"✅ Модель збережена в {model_path}")

# Завантаження моделі
loaded_model = LinearRegression(20)
checkpoint = torch.load(model_path, map_location=device)
loaded_model.load_state_dict(checkpoint['model_state_dict'])
loaded_model.eval()

print("✅ Модель успішно завантажена")

# Фінальне тестування
with torch.no_grad():
    test_input = torch.randn(5, 20)
    test_output = loaded_model(test_input)
    print(f"\nТест моделі - вхід: {test_input.shape}, вихід: {test_output.shape}")

print("\n" + "=" * 60)
print("🎉 НОУТБУК ЗАВЕРШЕНО УСПІШНО!")
print("=" * 60)