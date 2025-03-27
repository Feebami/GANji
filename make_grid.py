import PIL
import torch
import numpy as np
from torchvision import transforms as T
from torchvision.utils import make_grid
import os

files = os.listdir('kanji')
sample = np.random.choice(files, 9)
sample = [PIL.Image.open(f'kanji/{s}') for s in sample]
sample = [T.ToTensor()(s) for s in sample]
sample = torch.stack(sample)
grid = 255 - make_grid(sample, nrow=3) * 255
grid = grid.permute(1, 2, 0).numpy().astype('uint8')
grid = PIL.Image.fromarray(grid)
grid.save('display_imgs/input_kanji_sample.png')