import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Softmax
from keras.models import Sequential
from keras.datasets import mnist
from keras.callbacks import TensorBoard

import numpy as np
from pathlib import Path

def CNN(n_convset=5, n_dense=2, input_shape=(28, 28, 1), num_class=10):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation="relu", name="conv1"))
    model.add(Conv2D(32, (3, 3), activation="relu", name="conv2"))
    model.add(MaxPooling2D(name="mp1"))
    model.add(Conv2D(32, (3, 3), activation="relu", name="conv3"))
    model.add(Conv2D(32, (3, 3), activation="relu", name="conv4"))
    model.add(MaxPooling2D(name="mp2"))
    model.add(Flatten(name="ft1"))
    model.add(Dense(256, activation="relu", name="d1"))
    model.add(Dense(256, activation="relu", name="d2"))
    model.add(Dense(num_class, name="logits"))
    model.add(Softmax(name="softmax"))

    return model

if __name__ == "__main__":

    # 学習済みモデルの保存先
    model_dir = Path(".", "model_dir")
    model_dir.mkdir(exist_ok=True)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1) / 255.
    x_test = np.expand_dims(x_test, axis=-1) / 255.

    log_dir = Path("..", "logs")
    tb = TensorBoard(log_dir=str(log_dir))

    model = CNN()
    model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
    model.fit(x=x_train, y=y_train, epochs=10, validation_data=(x_test, y_test), callbacks=[tb])
    model.save(str(model_dir/"CNN.h5"))