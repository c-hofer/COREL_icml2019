# Connectivity-Optimized Representation Learning

This repository contains code to **reproduce** the experiments from

**Connectivity-Optimized Representation Learning via Persistent Homology**    
C. Hofer, R. Kwitt, M. Dixit and M. Niethammer    
*ICML '19*    
[PDF](http://proceedings.mlr.press/v97/hofer19a.html)

If you use this code (or parts of it), please cite this work as

```
@inproceedings{Hofer19a,
    title     = {Connectivity-Optimized Representation Learning via Persistent Homology},
    author    = {C.~Hofer, R.~Kwitt, M.~Dixit and M.~Niethammer},
    booktitle = {ICML},    
    year      = {2019}}
```

## Contents

- [Installation](#installation)
- [Datasets](#datasets)
- [Experiments](#experiments)

## Installation

The following setup was tested with the following system configuration:

- Ubuntu 18.04.2 LTS
- CUDA 10 (driver version 410.48)
- Anaconda (Python 3.7)
- PyTorch 1.1

In the following, we assume that we work in `/tmp` (obviously, you have to
	change this to reflect your choice and using `/tmp` is, of course, not
	the best choice :).

First, get the Anaconda installer and install Anaconda (in `/tmp/anaconda3`)
using

```bash
cd /tmp/
wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash Anaconda3-2019.03-Linux-x86_64.sh
# specify /tmp/anconda3 as your installation path
source /tmp/anaconda3/bin/activate
```

Second, we install PyTorch (v1.1) using

```bash
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

Third, we clone the `torchph` repository from GitHub (which basically
	implements all the functionality required for the experiments - previously named
	`chofer_torchex`) and make
	it available within Anaconda.

```bash
cd /tmp/
git clone https://github.com/c-hofer/torchph.git
cd torchph
git fetch --all --tags --prune     
git checkout tags/icml2019_code_release -b icml2019_code_release
cd ../
conda develop /tmp/torchph
```

Fourth, we clone this GitHub repository, using

```bash
cd /tmp/
git clone https://github.com/c-hofer/COREL_icml2019.git
cd COREL_icml2019
mkdir data
```

Finally, we modify `config.py` to reflect our choice of directories:

```python
ablation_bkb_dir = '/tmp/COREL_icml2019/models/ablation'
ablation_res_dir = '/tmp/COREL_icml2019/results_ablation'

performance_bkb_dir = '/tmp/COREL_icml2019/models/performance'
performance_res_dir = '/tmp/COREL_icml2019/results_performance'

dataset_root_generic = '/tmp/COREL_icml2019/data'
dataset_root_special = {}
```


## Datasets

Note that CIFAR10 and CIFAR100 are directly available via PyTorch and will
be downloaded automatically (to `/tmp/COREL_icml2019/data`). For TinyImageNet-200,
please use the following link and extract the downloaded zip file into
`/tmp/COREL_icml2019/data`:

```bash
cd /tmp/COREL_icml2019/data
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip http://cs231n.stanford.edu/tiny-imagenet-200.zip
```

*Note*: ImageNet instructions will be added soon!


## Experiments

All experiments have the same structure.
First a autoencoder, i.e., the "backbone", is trained on an auxiliary dataset, e.g., `CIFAR10`.
Then the trained backbone's encoder is used to represent samples from the test-dataset, e.g., `ImageNet`.

### Performance study

Here we train backbones on various auxiliary datasets (`CIFAR10`, `CIFAR100`, `TinyImageNet`) and evaluate the one-class performance on the test datasets (`CIFAR10`, `CIFAR100`, `TinyImageNet`, `ImageNet`).

```bash
cd /tmp/COREL_icml2019
python train_backbone_performance.py
python eval_backbone_performance.py
```

### Ablation study

In this group of experiments the overall impact of the hyper-parameters is evaluated.
Most importantly, the impact of the weighting factor of the proposed connectivity loss.

```bash
cd /tmp/COREL_icml2019
python train_backbone_ablation.py
python eval_backbone_ablation.py
```

### Fetching results

We provide two Jupyter notebooks to query results from the previous experiments,
in particular, `ablation_study.ipynb` and `performance_study.ipynb`.
