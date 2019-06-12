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
import matplotlib.pyplot as plt
from importlib import reload
from tqdm import tqdm
import time

import foolbox
from foolbox.models import TensorFlowModel

import model_zoo as zoo

class AdvImg:
    '''
    Adversarial Imageを作成するクラス
    '''
    def __init__(self, model):
        self.model = model
    
    def fgsm(self, x, y, epsilon=0.01, num_iter=100):
        '''
        FGSMを作るメソッド
        :Parameters:
        x: 入力画像 (None, 28, 28, 1)
        y: 正解ラベル (ex. "7" )
        epsilon: 摂動の強度
        num_iter: イテレーション回数
        :Returns:
        None
        '''
        self.x = x
        self.y = y
        self.preds = [] # イテレーションごとの推論結果の履歴
        print("True label: ", y)
        print(f"Initial pred: {np.argmax(self.model.predict(self.x), axis=-1)}")

        sess = K.get_session()

        print("セッション読み込み完了")
        self.x_adv = x # Adversarial Image
        self.noise = np.zeros_like(x) # ノイズ（摂動）
        self.adv_imgs = [] # Adversarial Imageの履歴

        for i in range(num_iter):
            # 損失を求める
            loss = K.sparse_categorical_crossentropy(self.y, self.model.output)
            
            # その損失に対して入力（画像）の勾配を求めて符号関数Signをかける
            grads = K.gradients(loss, self.model.input)[0]
            delta = K.sign(grads)

            # もともとのノイズ、Adversarial Imageに差分を追加して更新する
            self.noise = self.noise + epsilon*delta
            self.x_adv = self.x_adv + epsilon*delta
            self.x_adv = K.clip(self.x_adv, 0, 1) # 画像の表現範囲を超えないようにクリッピング

            # 計算グラフの入力に元画像を入れて計算グラフを実行。x_advの結果を取り出す。
            self.x_adv = sess.run(self.x_adv, feed_dict={self.model.input: self.x})
            self.preds.append(self.model.predict(self.x_adv))
            
            if (i+1) % 5 == 0:
                # 5イテレーションごとに結果を出力
                print(f"{i+1}, Pred: {np.argmax(self.preds[-1], axis=-1)}")
                self.adv_imgs.append(self.x_adv)

class AdvTraining:
    def __init__(self, kmodel, method, criterion):
        '''
        Adversarial Trainingを行うクラス
        params
        :kmodel: Kerasモデル（学習済み）
        :method: foolboxのbatch_attacksモジュール
        '''
        self.kmodel = kmodel
        self.fmodel = TensorFlowModel.from_keras(kmodel, bounds=(0, 1))
        self.criterion = criterion
        self.method_base = method
    
    def fit(self, x, y, epochs, validation_data=None, batch_size=32, verbose=1):
        '''
        params
        :x: オリジナルデータ
        :y: オリジナルラベル
        :i: イテレーション数
        '''
        self.x = x
        self.y = y
        for i in range(epochs):
            print(f'\n{i+1} / {epochs} epochs')
            # rand_idx = np.random.permutation(np.arange(self.x.shape[0])) # 全データのAdversarial Examplesをランダムな順番で作成
            rand_idx = np.random.choice(np.arange(self.x.shape[0]), size=batch_size) # 全データ作成している時間がないので、ランダムな10サンプル分を作成
            self.fmodel = TensorFlowModel.from_keras(self.kmodel, bounds=(0, 1))
            self.method = self.method_base(self.fmodel, criterion=self.criterion)

            self.adv_imgs, self.orig_cls = [], []
            for r_idx in rand_idx:
                x_adv = self.method(self.x[r_idx], self.y[r_idx], unpack=False)
                if x_adv.image is None:
                    continue
                self.adv_imgs.append(x_adv.image)
                self.orig_cls.append(x_adv.original_class)
            self.x = np.append(self.x, np.array(self.adv_imgs), axis=0)
            self.y = np.append(self.y, self.orig_cls)

            self.kmodel.fit(x=self.x, y=self.y, epochs=1, validation_data=validation_data, verbose=verbose)

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.
    x_test = x_test.reshape(-1, 28, 28, 1) / 255.

    model_path = Path("..", "model_dir")
    model_path.mkdir(exist_ok=True)
    model_path /= "CNN.h5"

    # 予め作成された学習モデルがmodel_dir/の中にあればロード、なければ学習を始める
    if model_path.exists():
        model = load_model(str(model_path))
    else:
        model = zoo.CNN(input_shape=(28, 28, 1)) # MNIST用
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
