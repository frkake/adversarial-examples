import numpy as np
import keras
from keras.models import load_model
from keras.datasets import mnist
from keras.callbacks import TensorBoard
import keras.backend as K
from collections import OrderedDict

import tensorflow as tf
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from importlib import reload

import model_make as mm

class AdvImg:
    def __init__(self, model):
        self.model = model
    
    def fgsm(self, x, y, epsilon=0.01, num_iter=100):
        self.x = x
        self.y = y
        self.preds = []
        print("True label: ", y)
        print(f"Initial pred: {np.argmax(self.model.predict(self.x), axis=-1)}")

        sess = K.get_session()

        print("セッション読み込み完了")
        self.x_adv = x
        self.noise = np.zeros_like(x)
        self.adv_his = []

        for i in range(num_iter):
            loss = K.sparse_categorical_crossentropy(self.y, self.model.output)
            
            grads = K.gradients(loss, self.model.input)[0]
            delta = K.sign(grads)

            self.noise = self.noise + delta
            self.x_adv = self.x_adv + epsilon*delta
            self.x_adv = K.clip(self.x_adv, 0, 1)

            self.x_adv = sess.run(self.x_adv, feed_dict={self.model.input: self.x})
            self.preds.append(self.model.predict(self.x_adv))
            
            if (i+1) % 10 == 0:
                print(f"{i+1}, Pred: {np.argmax(self.preds[-1], axis=-1)}")
                self.adv_his.append(self.x_adv)

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.

    model_path = Path("..", "model_dir")
    model_path.mkdir(exist_ok=True)
    model_path /= "CNN.h5"

    if model_path.exists():
        model = load_model(str(model_path))
    else:
        model = mm.CNN(input_shape=(28, 28, 1)) # MNIST用
        log_dir = Path("..", "logs")
        tb = TensorBoard(log_dir=str(log_dir))
        model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
        model.fit(x=x_train, y=y_train, epochs=5, validation_data=(x_test, y_test), callbacks=[tb])
        model.save(str(model_path)) 
    
    target_img = x_train[0:10].reshape(-1, 28, 28, 1)
    target_label = y_train[0:10]

    print("モデル読み込み完了")

    advgen = AdvImg(model)
    advgen.fgsm(x=target_img, y=target_label, epsilon=0.01, num_iter=100)
