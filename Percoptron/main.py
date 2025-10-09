import numpy as np

# Параметри
input_size = 25
hidden_size = 10
output_size = 4
learning_rate = 0.1
alpha = 0.01
threshold = 0.5
epochs = 100

# hidden_size = 10      

# 25 входів для кожної літери
letters = {
    "А": [
        0,1,1,1,0,
        1,0,0,0,1,
        1,1,1,1,1,
        1,0,0,0,1,
        1,0,0,0,1
    ],
    "Р": [
        1,1,1,1,0,
        1,0,0,0,1,
        1,1,1,1,0,
        1,0,0,0,0,
        1,0,0,0,0
    ],
    "В": [
        1,1,1,1,0,
        1,0,0,0,1,
        1,1,1,1,0,
        1,0,0,0,1,
        1,1,1,1,0
    ],
    "Л": [
        0,0,1,1,1,
        0,1,0,0,1,
        0,1,0,0,1,
        0,1,0,0,1,
        1,0,0,0,1
    ]
}

# One-hot кодування міток
label_encoding = {
    "А": [1, 0, 0, 0],
    "Р": [0, 1, 0, 0],
    "В": [0, 0, 1, 0],
    "Л": [0, 0, 0, 1]
}

# Формування навчальних даних
X_train = np.array([letters[ch] for ch in letters])
Y_train = np.array([label_encoding[ch] for ch in letters])


# Активації
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def step(x):
    return np.where(x >= threshold, 1, 0)


# Ініціалізація ваг
def init_weights():
    hidden_weights = np.random.uniform(-1, 1, (hidden_size, input_size))
    hidden_biases = np.random.uniform(-1, 1, hidden_size)
    output_weights = np.random.uniform(-1, 1, (output_size, hidden_size))
    output_biases = np.random.uniform(-1, 1, output_size)
    return hidden_weights, hidden_biases, output_weights, output_biases

# Навчання
def train_mlp(X, Y, method="reinforcement"):
    hidden_weights, hidden_biases, output_weights, output_biases = init_weights()

    for epoch in range(epochs):
        for i in range(len(X)):
            x = X[i]
            y = Y[i]

            # Forward pass
            hidden_input = np.dot(hidden_weights, x) + hidden_biases
            hidden_output = sigmoid(hidden_input)

            final_input = np.dot(output_weights, hidden_output) + output_biases
            final_output = step(final_input)

            error = y - final_output

            # Backward pass (тільки для reinforcement або alpha методів)
            if method == "reinforcement":
                for j in range(output_size):
                    if error[j] != 0:
                        delta_out = learning_rate * error[j]
                        output_weights[j] += delta_out * hidden_output
                        output_biases[j] += delta_out

                        # Оновлення прихованих ваг (спрощено)
                        delta_hidden = sigmoid_derivative(hidden_output) * (output_weights[j] * error[j])
                        hidden_weights += learning_rate * np.outer(delta_hidden, x)
                        hidden_biases += learning_rate * delta_hidden

            elif method == "alpha":
                for j in range(output_size):
                    if np.array_equal(final_output, y):
                        output_weights[j] += alpha * hidden_output
                        output_biases[j] += alpha
                    else:
                        delta_out = learning_rate * error[j]
                        output_weights[j] += delta_out * hidden_output
                        output_biases[j] += delta_out

                        delta_hidden = sigmoid_derivative(hidden_output) * (output_weights[j] * error[j])
                        hidden_weights += learning_rate * np.outer(delta_hidden, x)
                        hidden_biases += learning_rate * delta_hidden

    return hidden_weights, hidden_biases, output_weights, output_biases

# Тестування
def test_mlp(X, Y, hidden_weights, hidden_biases, output_weights, output_biases):
    correct = 0
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        hidden_output = sigmoid(np.dot(hidden_weights, x) + hidden_biases)
        final_output = step(np.dot(output_weights, hidden_output) + output_biases)
        if np.array_equal(final_output, y):
            correct += 1
    return correct / len(X)

# Навчання і тестування
results = {}
for method in ["reinforcement", "alpha"]:
    hw, hb, ow, ob = train_mlp(X_train, Y_train, method)
    acc = test_mlp(X_train, Y_train, hw, hb, ow, ob)
    results[method] = acc

print("Точність розпізнавання (MLP з прихованим шаром):")
for method, acc in results.items():
    print(f"Метод: {method} - {acc * 100:.2f}%")