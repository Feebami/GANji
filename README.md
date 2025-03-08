# GANji: real looking fake kanji
## Description
The purpose of this project was to explore various AI image generation technique's ability to recreate Japanese/Chinese Kanji. The methods tried were a Variational AutoEncoder, a Generative Adversarial Network, and a Denoising Diffusion Probabilistic Model. Each of these techniques were used to train various models on a dataset of about 10,000 images of standard font Kanji. 
## Table of Contents
- [Motivation](#motivation)
- [Datasets](#datasets)
- [Installation/Requirements](#installationrequirements)
- [Models](#models)
- [Usage](#usage)
- [Credits](#credits)
- [Further Reading](#further-reading)
## Motivation
The motivation for this project came from the desire to explore AI image generation techniques on a dataset that wouldn't be too unwieldy to work with. The dataset of Kanji was chosen because it was small enough to be trained on a personal computer, but complex enough to be interesting. The images being of small pixel size (48x48) and black and white made the dataset easy to manage and train many iterations of models on. Having studied Japanese Kanji in the past it was also easy to tell which images were good and which were bad.
## Datasets
- [Kanji Dataset](https://www.kaggle.com/datasets/frammie/kanjieast-asian-characters)
- Other test datasets
    - **MNIST**: The MNIST dataset is a dataset of 28x28 pixel images of handwritten digits. It is a common dataset used to test image classification algorithms.
    - **CIFAR-10**: The CIFAR-10 dataset is a dataset of 32x32 pixel images of 10 different classes of objects. It is a common dataset used to test image classification algorithms.
### Sample Kanji Data
![Sample Kanji Data](display_imgs/input_kanji_sample.png)
## Installation/Requirements
### Packages
- lightning=2.5.0
- matplotlib=3.10.0
- numpy=2.2.3
- python=3.11.11
- tk=8.6.13
- torch=2.6.0
- torchvision=0.21.0
- tqdm=4.67.1
### Installation
1. Clone the repository
2. Install the required packages
    - `pip install -r requirements.txt`
3. Download the Kanji dataset from the link above
## Models
### Variational AutoEncoder

## Usage
### Sample GANji
#### VAE
#### GAN
#### DDPM
## Credits
GAN related repositories:
[pytorch-spectral-normalization-gan](https://github.com/christiancosgrove/pytorch-spectral-normalization-gan)
[PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)
Diffusion related repositories:
[Diffusion-Models-pytorch](https://github.com/dome272/Diffusion-Models-pytorch/tree/main)
[denoising-diffusion-pytorch](https://github.com/rosinality/denoising-diffusion-pytorch/tree/master)
Triplet Attention:
[triplet-attention](https://github.com/landskape-ai/triplet-attention/tree/f07524e45db5eea1357c50316f30ab99a292d2f9)
## Further Reading
