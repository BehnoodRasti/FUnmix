# FUnmix
Fast Unmixing Using Alternating Method of Multipliers 

---

## Introduction

FUnmix is an open-source Python/PyTorch Package for Semi-supervised hyperspectral unmixing. It contains two algorithms called FaSUn and SUnS to solve the following nonconvex optimizations:

FaSUn (Fast Semissupervised Unmixing):
```math
  (\hat{\bf B},\hat{\bf A})=\arg\min_{{\bf B,A}} \frac{1}{2} || {\bf Y}-{\bf DBA}||_{F}^{2} ~~~
{\rm s.t.}~~~{\bf B}\geq 0,{\bf 1}_{m}^{T}{\bf B}={\bf 1}_{r}^{T},  {\rm and } ~~~ {\bf A}\geq 0,{\bf 1}_{r}^{T}{\bf A}={\bf 1}_{n}^{T}.
```
SUnS (Sparse Unmixing Using Soft-Shrinkage):
```math
 \hat{\bf X}=\arg\min_{{\bf X}} \frac{1}{2} || {\bf Y}-{\bf DX}||_{F}^{2}+\lambda ||{\bf X}||_1
~~~{\rm s.t.}~~~{\bf X}\geq 0,{\bf 1}_{m}^{T}{\bf X}={\bf 1}_{n}^{T},
```

Note: The provided tools can be used for signal and image processing applications beyond unmixing  such as source separation. 

## FUnmix Features

* Semisupervised category (Dictionary ${\bf D}$ should be provided)
* 2 unmixing methods (FaSUn, SUnS)
* 2 metrics (SRE, RMSE)
* 3 simulated datasets (located under `./data/`)

## License

FUnmix is distributed under MIT license.

## Citing FUnmix

Rasti, B., Zouaoui, A., Mairal, J., & Chanussot, J. (2024). Fast Semi-supervised Unmixing using Non-convex Optimization. [ArXiv. /abs/2308.09375](https://arxiv.org/abs/2401.12609)

## Installation

### Using `conda`

We recommend using a `conda` virtual Python environment to install FUnmix.

In the following steps we will use `conda` to handle the Python distribution and `pip` to install the required Python packages.
If you do not have `conda`, please install it using [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

```
conda create --name FUnmix python=3.10
```

Activate the new `conda` environment to install the Python packages.

```
conda activate FUnmix
```

Clone the Github repository.

```
git clone git@github.com:BehnoodRasti/FUnmix.git
```

Change directory and install the required Python packages.

```
cd FUnmix && pip install -r requirements.txt
```

## Important Note

The PyTorch was not included in the requirements.txt. You'll need to separately install PyTorch according to your OS and CUDA, please take a look https://pytorch.org/get-started/locally/. We tested the package on both Linux and Windows using Python 3.10 and pytorch-cuda=11.8.


## Getting started

This toolbox uses [MLXP](https://inria-thoth.github.io/mlxp/) to manage multiple experiments built on top of [hydra](https://hydra.cc/).

There are a few required parameters to define in order to run an experiment:
* `mode`: unmixing mode 
* `data`: hyperspectral unmixing dataset (DC1, DC2, and DC3)
* `model`: unmixing model (FaSUn or SUnS)
* `noise.SNR`: input SNR (*optional*)

An example of a corresponding command line is simply:

```shell
python main.py mode=semi data=DC1 model=FaSUn
```

## Data

### Data format

Datasets consist in a dedicated `.mat` file containing the following keys:

* `Y`: original hyperspectral image (dimension `p` x `n`)
* `E`: ground truth endmembers (dimension `p` x `r`)
* `A`: ground truth abundances (dimension `r` x `n`)
* `h`: HSI number of rows
* `w`: HSI number of columns
* `r`: number of endmembers
* `p`: number of channels
* `n`: number of pixels (`n` == `h`*`w`)

For sparse unmixing, a dictionary `D` containing `m` atoms is required.

* `D`: endmembers library (dimension `p` x `m`)
* `m`: number of atoms

## Parameter Tuning

### Fine Tuning

You may need to fine-tune the models' parameters for your application. Every method has a dedicated .yaml file located at config/model, which indicates the relevant parameters you can use for fine-tuning. For instance, for SUnCNN, the parameters are indicated in config/model/SUnCNN.yaml, and we can change the number of iterations and the input of the CNN with the following line. 

python unmixing.py mode=semi data=DC1 model=SUnCNN projection=True model.niters=8000 model.noisy_input=False noise.SNR=30

