import numpy as np

# Data Set
X = np.array([[0, 1, 1],
              [1, 1, 0],
              [1, 0, 0],
              [1, 1, 1],
              [0, 0, 1]])
y = np.array([[0], [0], [1], [1], [1]])

# Параметри нейронної мережі
input_dim = X.shape[1]  # Кількість входів
output_dim = 1          # Кількість виходів
lr = 0.1                # Швидкість навчання
epochs = 10000          # Кількість ітерацій

# Ініціалізація ваг і зміщення
weights = np.random.rand(input_dim, output_dim)
bias = np.random.rand(output_dim)

# Активаційна функція (сигмоїда)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Похідна сигмоїди
def sigmoid_derivative(x):
    return x * (1 - x)

# Навчання
for epoch in range(epochs):
    # Прямий прохід
    z = np.dot(X, weights) + bias
    predictions = sigmoid(z)

    # Обчислення похибки
    error = y - predictions

    # Зворотне поширення
    gradient = error * sigmoid_derivative(predictions)
    weights += np.dot(X.T, gradient) * lr
    bias += np.sum(gradient, axis=0) * lr

    # Лог для кожної 1000-ї ітерації
    if epoch % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Тестування
print("\nРезультати після навчання:")
for i, x in enumerate(X):
    result = sigmoid(np.dot(x, weights) + bias)
    print(f"Input: {x}, Predicted: {result[0]:.2f}, True: {y[i][0]}")
