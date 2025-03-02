from PIL import Image

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import messagebox
import torch
import torchvision

from GAN.resgan import Generator
from VAE.resnet_vae import Decoder

def load_model(model_type):
    if model_type == "GAN":
        # Load GAN model from checkpoint
        model = Generator(layer_list=[512, 256, 128])
        model.load_state_dict(torch.load('models/resnet_ganji_gen.pth', weights_only=False))
        model.eval()
    elif model_type == "VAE":
        # Load VAE model from checkpoint
        model = Decoder(latent_dim=256)
        model.load_state_dict(torch.load('models/resnet_ganji_decoder.pth'))
        model.eval()
    else:
        raise ValueError("Unknown model type")
    return model

def generate_kanji(model, model_type):
    # Generate Kanji images using the model
    # This is a placeholder function, replace with actual generation code
    print(f"Generating Kanji images using {model_type}")
    latent_dim = 128 if model_type == 'GAN' else 256
    num_images = 16
    latent = torch.randn(num_images, latent_dim)
    with torch.no_grad():
        images = model(latent)
    if model_type == 'GAN':
        images = images * 0.5 + 0.5
    else:
        images = torch.sigmoid(images)
    grid = torchvision.utils.make_grid(images, nrow=4)
    grid = 255 - grid * 255
    grid = grid.type(torch.uint8).cpu().numpy().transpose(1,2,0).squeeze()
    plt.figure(figsize=(10,10))
    plt.imshow(grid)
    plt.axis('off')
    plt.show()

def on_select():
    root.destroy()

    model_type = var.get()
    if model_type:
        model = load_model(model_type)
        generate_kanji(model, model_type)
    else:
        messagebox.showwarning("Selection Error", "Please select a model type")

# Create the main window
root = tk.Tk()
root.title("Select Model Type")

# Create a StringVar to hold the selected model type
var = tk.StringVar(value="")

# Create radio buttons for model selection
tk.Radiobutton(root, text="GAN", variable=var, value="GAN").pack(anchor=tk.W)
tk.Radiobutton(root, text="VAE", variable=var, value="VAE").pack(anchor=tk.W)

# Create a button to confirm the selection
tk.Button(root, text="Generate Kanji", command=on_select).pack()

# Run the application
root.mainloop()