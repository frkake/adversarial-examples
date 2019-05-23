import keras
from keras.datasets import mnist
import tensorflow as tf
import numpy as np

import importlib

class AdversarialImage:
    def __init__(self, model_path, num_class=10):
        self.model = keras.models.load_model(model_path)
        self.num_class = num_class

        self.model.summary()
    
    def step_fgsm(self, x, epsilon, logits, label):
        onehot_label = tf.one_hot(label, self.num_class)
        crossentropy = tf.losses.softmax_cross_entropy(onehot_label, logits, label_smoothing=0.1, weights=1.)
        x_adv = x + epsilon * tf.sign(tf.gradients(crossentropy, x)[0])
        x_adv = tf.clip_by_value(x_adv, -1, 1)
        x_adv = tf.stop_gradient(x_adv)

        return x_adv

