from __future__ import division

import torch
import torch.nn as nn
import logging
import numpy as np
import os
from torch import nn, optim
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import SpectralAngleMapper, PeakSignalNoiseRatio
from torchmetrics.image.fid import FrechetInceptionDistance
from spectral import *
import cv2
from NTIRE2022Util import *
import scipy.io
from utils import *
import matplotlib.pyplot as plt
from PIL import Image

def visualize_result(name, root, hsi):
    sample = scipy.io.loadmat(root + str(hsi).zfill(3) + '.mat')
    hyper = sample['cube']
    rgb = sample['rgb']
    filter, _ = load_rgb_filter('resources/CIE_xyz_1964_10deg.csv')
    try:
        os.mkdir(f'results/{name}/')
    except:
        pass
    images = []
    bands = [400, 500, 550, 700]
    b = 0

    images.append(rgb)
    for band in bands:
        image = np.zeros([128, 128, 3])
        # Assuming 'filter' and 'hyper' are defined
        for i in range(128):
            for j in range(128):
                image[i, j, :] = filter[band - 360, :] * hyper[i, j, b]
        image = (image - image.min()) / (image.max() - image.min())
        images.append(image)

    # Calculate the total height of the result image including gaps
    gap_height = 10
    result_height = 128 * len(images) + gap_height * (len(images) - 1)

    # Create a new blank image with white background
    result_image = Image.new('RGB', (128, result_height), color='white')

    # Paste each individual image into the result with 10 pixels gap
    for idx, image in enumerate(images):
        pil_image = Image.fromarray((image * 255).astype('uint8'))
        result_image.paste(pil_image, (0, (128 + gap_height) * idx))

    # Add a title below the combined image
    title = name
    plt.figure(figsize=(8, 8))
    plt.imshow(result_image)
    plt.title(title, fontsize=20, pad=10)
    plt.axis('off')
    plt.savefig(f'results/{name}/{hsi}.png', format='PNG', bbox_inches='tight', pad_inches=0, transparent=True, dpi=300)

    # Open and save the result image again using PIL to remove compression
    img = Image.open(f'results/{name}/{hsi}.png')
    img.save(f'results/{name}/{hsi}.png', format='PNG', compress_level=0)

if __name__ == '__main__':
    root = '/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/SNCWGANunet/'
    hsis = [5642,5453,5621]
    for hsi in hsis:
        visualize_result('SNCWGAN+Unet', root, hsi)