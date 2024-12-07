import numpy as np
import matplotlib.pyplot as plt

# Третя задача: прогнозування Time Series

# Генерація синтетичного набору даних
def generate_time_series(n_steps):
    time = np.linspace(0, 50, n_steps)
    series = np.sin(time) + 0.1 * np.random.randn(n_steps)  # Синусоїда з шумом
    return time, series

n_steps = 100
look_back = 10
future_steps = 10

time, series = generate_time_series(n_steps)

# Підготовка даних для навчання
X, y = [], []
for i in range(len(series) - look_back):
    X.append(series[i:i+look_back])
    y.append(series[i+look_back])

X = np.array(X)
y = np.array(y).reshape(-1, 1)

# Нормалізація даних
X_mean, X_std = X.mean(), X.std()
X = (X - X_mean) / X_std
y_mean, y_std = y.mean(), y.std()
y = (y - y_mean) / y_std

# Параметри нейронної мережі
input_dim = X.shape[1]
hidden_dim = 16
output_dim = 1
lr = 0.01
epochs = 500

# Ініціалізація ваг
W1 = np.random.rand(input_dim, hidden_dim)
W2 = np.random.rand(hidden_dim, output_dim)
b1 = np.random.rand(hidden_dim)
b2 = np.random.rand(output_dim)

# Активаційна функція
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Навчання
for epoch in range(epochs):
    # Прямий прохід
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    predictions = Z2  # Лінійний вихід

    # Обчислення похибки
    error = y - predictions

    # Зворотний прохід
    dZ2 = -2 * error
    dW2 = np.dot(A1.T, dZ2) / len(X)
    db2 = np.sum(dZ2, axis=0) / len(X)

    dZ1 = np.dot(dZ2, W2.T) * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / len(X)
    db1 = np.sum(dZ1, axis=0) / len(X)

    # Оновлення ваг
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    # Лог кожної 50-ї ітерації
    if epoch % 50 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Прогнозування
X_test = series[-look_back:]
X_test = (X_test - X_mean) / X_std  # Нормалізація
predictions = []
for _ in range(future_steps):
    Z1 = np.dot(X_test, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    next_value = Z2[0]
    predictions.append(next_value)
    X_test = np.roll(X_test, -1)
    X_test[-1] = next_value

# Денормалізація прогнозів
predictions = np.array(predictions) * y_std + y_mean

# Візуалізація
plt.figure(figsize=(10, 6))
plt.plot(time, series, label="Actual Series")
plt.plot(time[-future_steps:], predictions, label="Predicted", linestyle="--")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Time Series Prediction")
plt.show()
