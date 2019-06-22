# COREL_icml2019
Code for "Connectivity-Optimized Representation Learning via Persistent Homology", ICML 2019

# Installation 

Prerequisites: 
	Anaconda (installer: Anaconda3-2019.03-Linux-x86_64.sh)
	Pytorch 1.1 (python 3.7, cuda 10)


```
conda develop path/to/your/chofer_torchex/clone

```

#Experiment

All experiments have the same structure. 
First a autoencoder, i.e., "backbone", is trained on an auxiliary dataset, e.g., `CIFAR10`.
Then the trained backbone's encoder is used to represent samples from the test-dataset, e.g., `ImageNet`. 


*performance study*: here we train backbones on various auxiliary datasets (`CIFAR10`, `CIFAR100`, `TinyImageNet`) and evaluate the one-class performance on the test datasets (`CIFAR10`, `CIFAR100`, `TinyImageNet`, `ImageNet`). 

*ablation study*: in this group of experiments the overall impact of the hyper-parameters is evaluated. 
Most importantly, the impact of the weighting factor of the proposed connectivity loss. 

## Run experiments

*Backbone training*. First we have to train the various backbones
```
python train_backbone_performance.py
python train_backbone_ablation.py
```