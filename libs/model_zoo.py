import keras
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Softmax, InputLayer, Conv2DTranspose, UpSampling2D
from keras.models import Sequential
from keras.datasets import mnist
from keras.callbacks import TensorBoard

import numpy as np
from pathlib import Path

def CNN(n_convset=3, n_dense=2, input_shape=(28, 28, 1), num_class=10):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation="relu", name="conv0", input_shape=input_shape, padding='same'))
    for i in range(n_convset):
        model.add(Conv2D(32, (3, 3), activation="relu", name=f"conv{i}-1", padding='same'))
        model.add(Conv2D(32, (3, 3), activation="relu", name=f"conv{i}-2", padding='same'))
        model.add(MaxPooling2D(name=f"mp{i}"))
    model.add(Flatten(name="ft1"))

    for i in range(n_dense):
        model.add(Dense(256, activation="relu", name=f"d{i}"))
    
    model.add(Dense(num_class, name="logits"))
    model.add(Softmax(name="softmax"))

    return model


def ConvDAE(input_shape=(28, 28, 1)):
    model = Sequential()

    # Encoder
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # (14, 14, 1)
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # (7, 7, 1)

    # Decoder
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D()) # (14, 14, 64)
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D()) # (28, 28, 1)
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

    return model

if __name__ == "__main__":

    # 学習済みモデルの保存先
    model_dir = Path("..", "model_dir")
    model_dir.mkdir(exist_ok=True)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1) / 255.
    x_test = np.expand_dims(x_test, axis=-1) / 255.

    train_noise = np.abs(np.random.normal(0.0, 0.2, x_train.shape))

    test_noise = np.abs(np.random.normal(0.0, 0.2, x_test.shape))

    x_train_noised = np.clip(x_train + train_noise, 0.0, 1.0)
    x_test_noised = np.clip(x_test + test_noise, 0.0, 1.0)

    log_dir = Path("..", "logs")
    tb = TensorBoard(log_dir=str(log_dir))

    model = ConvDAE()
    model.compile(optimizer="Adam", loss=keras.losses.MSE, metrics=[keras.metrics.mse])
    model.fit(x=x_train_noised, y=x_train, epochs=5, validation_data=(x_test_noised, x_test), callbacks=[tb])
    model.save(str(model_dir/"ConvDAE.hdf5"))

