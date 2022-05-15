# Deep Image Clustering Based on Contrastive Learing and Graph Convolutional Networks  


This repo contains the  implementation of our paper
## requirement
The first stage code is based on pytorch, and second stage code is based on tensorflow.

```shell
pytroch >= 1.6
tensorflow >= 2.3.1
faiss-gpu
spektral = 1.0.6
tensorboardX
```


## Training(first stage)

### Train model
For example, run the following commands sequentially  on CIFAR10:
```shell
python simclr.py --config_env configs/your_env.yml --config_exp configs/pretext/simclr_cifar10.yml
python test.py --config_env configs/env.yml --config_exp configs/pretext/simclr_cifar10.yml
```
After that, we will get image feature file named as cifar10-features.npy. And the true labels file named as cifar10-lables.npy. The true labels are used for calculating acc, nmi, and ari.



## Training(second stage)
You should move the feature from first stage to GCN-clustering/data folder.

The details are marked in the jupynotebook files clustering.ipynb.

And the tensorboard logdir will be in result_cifar10 folder.

### Remark

We need to point out that in the paper we use the GCN layers, but in our code we used a skip connected GCN layers to replace the self-loop.







