# Generating Adversarial Examples by Keras and TensorFlow

TensorFlowバックエンドKerasを使ったAdversarial Examplesの生成リポジトリです。  
基本的にできるだけMNISTでAdversarial Examples作ろうと思っています。  
MNISTだと摂動が非常わかりやすので、結果の確認のためにそちらのほうがいいと判断しています。

- 自前で作ったFGSM Keras版 ([keras-mnist.ipynb](./notebooks/keras-mnist.ipynb))
- 自前で作ったベーシックな攻撃・FGSM PyTorch版（[pytorch-mnist.ipynb](./notebooks/pytorch-mnist.ipynb)）  
  ターゲット攻撃があってるのか自身薄...
- Cleverhansを使ったFGSM ([cleverhans-fgsm-tensorflow.ipynb](./notebooks/cleverhans-fgsm-tensorflow.ipynb))
- foolboxを使った様々な攻撃 ([foolbox-fgsm-keras.ipynb](./notebooks/foolbox-fgsm-keras.ipynb))
- DAEを使ったAdversarial Examplesの防御([DAE_defence.ipynb](./notebooks/DAE_defence.ipynb))