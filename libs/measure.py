import tensorflow as tf
import numpy as np
from skimage.measure import compare_psnr, compare_ssim

class TFMetrics:
    '''Tensorflow演算用'''
    def __init__(self):
        pass
    
    @staticmethod
    def psnr(y_true, y_pred):
        return tf.image.psnr(y_pred, y_true, max_val=1.0)

    @staticmethod
    def ssim(y_true, y_pred):
        return tf.image.ssim(y_pred, y_true, max_val=1.0)

    @staticmethod
    def distortion(y_true, y_pred):
        '''歪み尺度'''
        x = tf.math.subtract(y_pred, y_true)
        x = tf.math.pow(x, 2)
        x = tf.math.reduce_sum(x)
        x = tf.math.divide(x, tf.cast(tf.size(y_true), tf.float32))
        x = tf.math.sqrt(x)
        return x

class NpMetrics:
    '''Numpy配列用'''
    def __init__(self):
        pass
    
    @staticmethod
    def psnr(y_true, y_pred, data_range):
        return compare_psnr(y_true, y_pred, data_range=data_range)
    
    @staticmethod
    def ssim(y_true, y_pred, data_range):
        return compare_ssim(y_true, y_pred, data_range=data_range)
    
    @staticmethod
    def distortion(y_true, y_pred):
        '''歪み尺度'''
        x = y_pred - y_true
        x = x**2
        x = np.sum(x)
        x = x / y_true.size
        x = np.sqrt(x)
        return x

if __name__ == '__main__':
    pass