# Generating Adversarial Examples by Keras and TensorFlow
TensorFlowバックエンドKerasを使ったAdversarial Examplesの生成リポジトリです。  
基本的にできるだけMNISTでAdversarial Examplesつ作ろうと思っています。
MNISTだと摂動が非常わかりやすのので  

- Cleverhansを使ったFGSM ([cleverhans-fgsm-tensorflow.ipynb](notebook/cleverhans-fgsm-tensorflow.ipynb))
- foolboxを使ったFGSM ([foolbox-fgsm-keras.ipynb](notebook/foolbox-fgsm-keras.ipynb))
- 自前で作ったFGSM Keras版 ([fgsm_mnist.ipynb](notebook/fgsm_mnist.ipynb))