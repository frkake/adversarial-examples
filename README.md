# Generating Adversarial Examples by Keras and TensorFlow

TensorFlowバックエンドKerasを使ったAdversarial Examplesの生成リポジトリです。  
基本的にできるだけMNISTでAdversarial Examples作ろうと思っています。  
MNISTだと摂動が非常わかりやすので、結果の確認のためにそちらのほうがいいと判断しています。

- 自前で作ったFGSM Keras版 ([keras-mnist.ipynb](./notebooks/keras-mnist.ipynb))
- 自前で作ったベーシックな攻撃、FGSM PyTorch版（[pytorch-mnist.ipynb](./notebooks/pytorch-mnist.ipynb)）
- Cleverhansを使ったFGSM ([cleverhans-fgsm-tensorflow.ipynb](./notebooks/cleverhans-fgsm-tensorflow.ipynb))
- foolboxを使ったFGSM ([foolbox-fgsm-keras.ipynb](./notebooks/foolbox-fgsm-keras.ipynb))