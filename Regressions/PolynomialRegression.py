import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def generate_data(count):
   np.random.seed(42)
   x = np.random.rand(count, 1) # random between 0 and 1
   y = get_y_value(x) + 0.1 * np.random.randn(count, 1)
   return x, y


def get_y_value(x):
    return 2 * np.power(x, 2) + x + 1


def build_model():
   model = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(1,)),
      tf.keras.layers.Dense(units=16, activation='relu'),
      tf.keras.layers.Dense(units=16, activation='relu'),
      tf.keras.layers.Dense(units=1, activation='linear')
   ])

   model.compile(optimizer='sgd', loss='mean_squared_error')
   return model


def plot(x, y, predictions):
    plt.scatter(x, y, label='Correct Data', color='black')
    plt.scatter(x, predictions, label='Predictions', color='green', marker='x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Predictions')
    plt.legend()
    plt.show()


def train_model(model):
    x, y = generate_data(2500)
    model.fit(x, y, epochs=100, verbose=0)


def get_trained_model():
    model = build_model()
    train_model(model)
    return model
    

def main():
    model  = get_trained_model()
    x_test = np.arange(0.1, 1, 0.1)
    y_test = get_y_value(x_test)

    predictions = model.predict(x_test).flatten()
    plot(x_test, y_test, predictions)


if __name__ == "__main__":
   print()
   main()
   print()
