
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import colour
import scipy.io
import numpy as np
import os

resultpath = 'result/'
model_name = 'SNCWGAN/'
root = '/work3/s212645/Spectral_Reconstruction/checkpoint/'
metrics = 'metrics.pth'
path = root + model_name + metrics
result = torch.load(path, map_location=torch.device('cpu'))

MRAE = result['MRAE']
RMSE = result['RMSE']
PSNR = result['PSNR']
SAM = result['SAM']
length = len(MRAE)
for i in range(length):
    MRAE[i] = MRAE[i].item()
for i in range(length):
    RMSE[i] = RMSE[i].item()
for i in range(length):
    PSNR[i] = PSNR[i].item()
for i in range(length):
    SAM[i] = SAM[i].item()

plt.figure(figsize=[20,5])
plt.subplot(1, 4, 1)
plt.plot(range(length), MRAE)
plt.title("MRAE")

plt.subplot(1, 4, 2)
plt.plot(range(length), RMSE)
plt.title("RMSE")

plt.subplot(1, 4, 3)
plt.plot(range(length), PSNR)
plt.title("PSNR")

plt.subplot(1, 4, 4)
plt.plot(range(length), SAM)
plt.title("SAM")
plt.close()


def MRAE(x,y):
    return (x-y) ** 2

# realroot = '/work3/s212645/Spectral_Reconstruction/RealHyperSpectrum/SNCWGAN/'
# fakeroot = '/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/SNCWGAN/'
# realroot = '/work3/s212645/Spectral_Reconstruction/RealHyperSpectrum/D2GAN/'
# fakeroot = '/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/D2GAN/'
# realroot = '/work3/s212645/Spectral_Reconstruction/RealHyperSpectrum/SNCWGANNoNoise/'
# fakeroot = '/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/SNCWGANNoNoise/'
realroot = '/work3/s212645/Spectral_Reconstruction/RealHyperSpectrum/SNCWGANNoNoise/'
fakeroot = '/work3/s212645/Spectral_Reconstruction/FakeHyperSpectrum/SNCWGANNoNoise/'
for k in range(100):
    num = str(k).zfill(3)
    name = num + '.mat'
    savepath = resultpath + num + '_Figures/'
    try:
        os.mkdir(savepath)
    except:
        pass

    fakergb = scipy.io.loadmat(fakeroot + name)['rgb']
    fakergb = (fakergb - fakergb.min()) / (fakergb.max()-fakergb.min())

    realrgb = scipy.io.loadmat(realroot + name)['rgb']
    realrgb = (realrgb - realrgb.min()) / (realrgb.max()-realrgb.min())

    plt.figure(figsize=[12,6])
    plt.subplot(1, 2, 1)
    plt.imshow(realrgb)
    plt.title('Ground Truth')
    plt.subplot(1, 2, 2)
    plt.imshow(fakergb)
    plt.title('Fake Generation')
    plt.savefig(savepath+'Reconstructed_RGB_image.png')
    plt.close()
    
    fake = scipy.io.loadmat(fakeroot + name)['cube']
    real = scipy.io.loadmat(realroot + name)['cube']
    image1_lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(realrgb))
    image2_lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(fakergb))
    colour.delta_E(image1_lab, image2_lab).mean()
    deltae = colour.delta_E(image1_lab, image2_lab)
    sns.heatmap(deltae, cmap='jet')

    colour.delta_E(image1_lab, image2_lab, method="CIE 1976").mean()
    deltae = colour.delta_E(image1_lab, image2_lab, method="CIE 1976")
    sns.heatmap(deltae, cmap='jet')
    plt.savefig(savepath+'DeltaEHeatmap.png')
    plt.close()

    colour.delta_E(image1_lab, image2_lab, method="CIE 2000")
    deltae = colour.delta_E(image1_lab, image2_lab, method="CIE 2000")
    sns.heatmap(deltae, cmap='jet')


    COLOR = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
    COLOR = sns.color_palette("light:b", as_cmap=True)
    bands = np.linspace(400, 700, 31)
    for i in range(31):
        band = bands[i]
        data = MRAE(real, fake)[:,:,i]
        # plt.subplot(6, 6, i+1)
        sns.heatmap(data, cmap='jet')
        plt.title('MRAE Hearmap Band: {}'.format(band))
        filename = savepath+'MRAEHeatmap_'+str(band)+'.png'
        plt.savefig(filename)
        plt.close()

    def SAMHeatMap(preds, target):
        dot_product = np.sum(preds * target, axis=2)
        preds_norm = np.linalg.norm(preds, axis=2)
        target_norm = np.linalg.norm(target, axis=2)
        sam_score = np.arccos(dot_product / (preds_norm * target_norm))
        return sam_score

    sam = SAMHeatMap(fake, real)

    sns.heatmap(sam, cmap='jet')
    filename = savepath+'SAMHeatmap.png'
    plt.savefig(filename)
    plt.close()
