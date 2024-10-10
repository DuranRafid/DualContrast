# DualContrast: Unsupervised Disentangling of Content and Transformations with Implicit Parameterization

Unsupervised disentanglement of content and transformation has recently drawn much research, given their efficacy in solving downstream unsupervised tasks like clustering, alignment, and shape analysis. This problem is particularly important for analyzing shape-focused real-world scientific image datasets, given their significant relevance to downstream tasks. The existing works address the problem by explicitly parameterizing the transformation factors, significantly reducing their expressiveness. Moreover, they are not applicable in cases where transformations can not be readily parametrized. An alternative to such explicit approaches is self-supervised methods with data augmentation, which implicitly disentangles transformations and content. We demonstrate that the existing self-supervised methods with data augmentation result in the poor disentanglement of content and transformations in real-world scenarios. Therefore, we developed a novel self-supervised method, DualContrast, specifically for unsupervised disentanglement of content and transformations in shape-focused image datasets. Our extensive experiments showcase the superiority of DualContrast over existing self-supervised and explicit parameterization approaches. We leveraged DualContrast to disentangle protein identities and protein conformations in cellular 3D protein images. Moreover, we also disentangled transformations in MNIST, viewpoint in the Linemod Object dataset, and human movement deformation in the Starmen dataset as transformations using DualContrast.

This repository contains the code and instructions to train and evaluate DualContrast on MNIST, LineMod, Starmen, and the 3D subtomogram dataset.

## Package Requirements

- Pytorch
- TorchVision
- Numpy
- Pillow

When instaling the above packages, please make sure the versions of them are consistent with each other. We used a pytorch version of 1.9.0, torchvision version of 0.10.0, numpy version of 1.24.4, Pillow version of 8.2.0. However, the code should also work fine on newer version of these packages.

For subtomogram dataset, we used the following additional package:

- Kornia 

We used version 0.5.4 of kornia package. 

## Training and Evaluating DualContrast on MNIST

For reproducing our results on MNIST dataset, simply run the following command:

```
python train_mnist.py -eval-only
```

This should load the model we provided and produce the results in our manuscript.

For training the model, saving it, and evaluating the trained model qualitatively and quantitatively, simply run the following command:

```
python train_mnist.py
```

If you do not want to use the default values of the hyperparameters and rather use your own setting, use the following:
```
python train_mnist.py --num-epochs <number of epochs> --z-dim <dimension of the latent factors> --batch-size <Batch size for training> --learning-rate <learning-rate for training>
```

If you want to only train and save a model without evaluating, run the following:

```
python train_mnist.py -train-only --num-epochs <number of epochs> --z-dim <dimension of the latent factors> --batch-size <Batch size for training> --learning-rate <learning-rate for training>
```

## Training and Evaluating DualContrast on LINEMOD

Download the 'lm_train.zip' file from [this url](https://bop.felk.cvut.cz/media/data/bop_datasets/lm_train.zip) inside the data folder. Unzip the folder. The run the following command to create the dataset for DualContrast.
 
 
```
python data/create-linemod-dataset.py
```

For reproducing our results on Linemod dataset, simply run the following command:

```
python train_linemod.py -eval-only
```

This should load the model we provided and produce the results in our manuscript.

For training the model, saving it, and evaluating the trained model qualitatively and quantitatively, simply run the following command:

```
python train_linemod.py
```

If you do not want to use the default values of the hyperparameters and rather use your own setting, use the following:
```
python train_linemod.py --num-epochs <number of epochs> --z-dim <dimension of the latent factors> --batch-size <Batch size for training> --learning-rate <learning-rate for training>
```

If you want to only train and save a model without evaluating, run the following:

```
python train_linemod.py -train-only --num-epochs <number of epochs> --z-dim <dimension of the latent factors> --batch-size <Batch size for training> --learning-rate <learning-rate for training>
```


## Training and Evaluating DualContrast on Starmen

Download the 'starmen.tar.gz' file from [this zenodo url](https://zenodo.org/records/5081988) inside the data folder. Unzip the folder. The run the following command to create the dataset for DualContrast.
 
 
```
python data/create-starmen-dataset.py
```

For reproducing our results on Starmen dataset, simply run the following command:

```
python train_starmen.py -eval-only
```

This should load the model we provided and produce the results in our manuscript.

For training the model, saving it, and evaluating the trained model qualitatively, simply run the following command:

```
python train_starmen.py
```

If you do not want to use the default values of the hyperparameters and rather use your own setting, use the following:
```
python train_starmen.py --num-epochs <number of epochs> --z-dim <dimension of the latent factors> --batch-size <Batch size for training> --learning-rate <learning-rate for training>
```

If you want to only train and save a model without evaluating, run the following:

```
python train_starmen.py -train-only --num-epochs <number of epochs> --z-dim <dimension of the latent factors> --batch-size <Batch size for training> --learning-rate <learning-rate for training>
```

## Training and Evaluating DualContrast on Subtomogram Dataset

Download the 'composition_conformation_subtomograms.pkl' file from [this zenodo url](https://zenodo.org/records/11244440) inside the data folder. 


For training the model, saving it, and save the embedding, simply run the following command:

```
python train_subtomo.py
```

If you do not want to use the default values of the hyperparameters and rather use your own setting, use the following:
```
python train_subtomo.py --num-epochs <number of epochs> --z-dim <dimension of the latent factors> --batch-size <Batch size for training> --learning-rate <learning-rate for training>
```

If you want to only train and save a model without saving the embedding, run the following:

```
python train_subtomo.py -train-only --num-epochs <number of epochs> --z-dim <dimension of the latent factors> --batch-size <Batch size for training> --learning-rate <learning-rate for training>
```

If you want to only obtain content and transformation embeddings from a saved model without training it, run the following:

```
python train_subtomo.py -eval-only
```
But make sure to save the model in the models folder.


