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
## Requirements
- python=3.11
### Packages
- lightning=2.5.0
- matplotlib=3.10.0
- numpy=2.2.3
- torch=2.6.0
- torchvision=0.21.0
- tqdm=4.67.1
## Pipeline
### Data Loading
The Kanji images were loaded into memory using Python Imaging Library (PIL) and torchvision transformations. The images were resized to 64 by 64 pixels to allow for more pooling in the convolutional networks. A random horizontal flip was used as data augmentation to effectively double the training dataset. Although Kanji characters are generally not horizontally symmetrical, the benefit of increased dataset size outweighed potential distortions. This preprocessing step enhanced the model's ability to generalize by providing varied training inputs.
### Models
#### Variational AutoEncoder
The Variational AutoEncoder (VAE) consists of two primary components: an encoder, which compresses input images into a compact n-dimensional latent representation, and a decoder, which reconstructs the original images from these compressed representations. The model is optimized by minimizing a combined loss function that consists of two terms: the binary cross-entropy loss between the original and reconstructed pixel values, and the Kullback-Leibler divergence of the learned latent space distribution from a predefined prior distribution.

The final VAE architecture utilized a ResNet-like design. Specifically, the encoder and decoder each comprised 7 residual blocks, with the encoder progressively increasing channel dimensions from 64 to 512 filters, and the decoder symmetrically decreasing them from 512 back down to 64. These residual blocks were combined with convolutional layers at the input and output stages to handle the specified image dimensions, as well as linear layers to transition between the three-dimensional image tensors and the lower-dimensional latent space.

See [resnet_vae.py](VAE/resnet_vae.py) for model creation code.
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
