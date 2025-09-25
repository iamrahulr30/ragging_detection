import tensorflow as tf
import numpy as np

# 1. Compute gradients for expressions
w1 = tf.Variable(3.0)
w2 = tf.Variable(3.0)
with tf.GradientTape() as tape:
    # expr1 = 3 * w1**2 + 2 * w1 * w2
    expr2 = w1**2
gradients = tape.gradient([expr1], [w1, w2])
print("Gradients for expr1 and expr2:", gradients[0].numpy(), gradients[1].numpy())

# 2. Binary cross-entropy (NumPy formula)
y_true = np.array([0, 1])
y_pred = np.array([0.5, 0.9])
bce_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
print("Binary cross-entropy loss:", bce_loss)

# 3. Neural network with one iteration
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

X = np.array([[1, 2], [4, 5]])
y = np.array([[3], [6]])
model = Sequential([
    Dense(2, activation='linear', input_shape=(2,), kernel_initializer='zeros', bias_initializer='zeros'),
    Dense(1, activation='linear', kernel_initializer='zeros', bias_initializer='zeros')
])
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss='mean_squared_error')
model.train_on_batch(X, y)

for i, layer in enumerate(model.layers, 1):
    weights, biases = layer.get_weights()
    print(f"Layer {i} weights:", weights)
    print(f"Layer {i} biases:", biases)
