import numpy as np

# Друга задача: ідентифікація бінарних зображень
# Приклад Data Set (матриця растра для символів)
data = {
    'A': np.array([1, 1, 1, 1, 0, 1, 1, 0, 1]),
    'B': np.array([1, 1, 1, 1, 1, 0, 1, 1, 1]),
    'C': np.array([1, 1, 1, 1, 0, 0, 1, 1, 1]),
}

# Перетворення Data Set у формат навчання
X = np.array(list(data.values()))  # Входи
labels = list(data.keys())
y = np.eye(len(labels))  # One-hot енкодинг для виходів

# Параметри нейронної мережі
input_dim = X.shape[1]
hidden_dim = 6
output_dim = y.shape[1]
lr = 0.1
epochs = 10000

# Ініціалізація ваг
W1 = np.random.rand(input_dim, hidden_dim)
W2 = np.random.rand(hidden_dim, output_dim)
b1 = np.random.rand(hidden_dim)
b2 = np.random.rand(output_dim)

# Активаційна функція
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Навчання
for epoch in range(epochs):
    # Прямий прохід
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    predictions = sigmoid(Z2)

    # Обчислення похибки
    error = y - predictions

    # Зворотний прохід
    dZ2 = error * sigmoid_derivative(predictions)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0)

    dZ1 = np.dot(dZ2, W2.T) * sigmoid_derivative(A1)
    dW1 = np.dot(X.T, dZ1)
    db1 = np.sum(dZ1, axis=0)

    # Оновлення ваг
    W1 += lr * dW1
    b1 += lr * db1
    W2 += lr * dW2
    b2 += lr * db2

    if epoch % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Перевірка
print("\nРезультати після навчання:")
for i, x in enumerate(X):
    Z1 = np.dot(x, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    output = sigmoid(Z2)
    predicted_label = labels[np.argmax(output)]
    print(f"Input: {x}, Predicted: {predicted_label}, True: {labels[i]}")
