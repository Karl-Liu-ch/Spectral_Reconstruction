import sys
sys.path.append('./')
from utils import reconRGBfromNumpy
import numpy as np
import scipy.io
import os
import h5py

root = '/work3/s212645/Spectral_Reconstruction/'
ARAD = 'ARAD/'
BGU1 = '/work3/s212645/Spectral_Reconstruction/BGU_HS/NTIRE2018_Train1_Spectral/'
BGU2 = '/work3/s212645/Spectral_Reconstruction/BGU_HS/NTIRE2018_Train2_Spectral/'
BGU3 = '/work3/s212645/Spectral_Reconstruction/BGU_HS/NTIRE2018_Validate_Spectral/'

cleanpath = '/work3/s212645/Spectral_Reconstruction/clean/'
# os.mkdir(cleanpath)
# os.mkdir(cleanpath + ARAD)
# os.mkdir(cleanpath + 'BGU/')

# for i in range(950):
#     matpath = str(i + 1).zfill(3) + '.mat'
#     matpath = root + ARAD + matpath
#     mat = scipy.io.loadmat(matpath)
#     hyper = mat['cube']
#     rgbrecon = reconRGBfromNumpy(hyper)
#     mat['rgb'] = rgbrecon
#     savepath = cleanpath + ARAD + str(i + 1).zfill(3) + '.mat'
#     scipy.io.savemat(savepath, mat)

# for i in range(203):
#     matpath = str(i + 1).zfill(3) + '.mat'
#     matpath = BGU1 + 'BGU_HS_00' + matpath
#     Mat = {}
#     with h5py.File(matpath, 'r') as mat:
#         hyper = np.transpose(mat.get('rad'), [2,1,0])
#         hyper = (hyper-hyper.min())/(hyper.max()-hyper.min())
#         Mat['cube'] = hyper
#         rgbrecon = reconRGBfromNumpy(hyper)
#         Mat['rgb'] = rgbrecon
#         savepath = cleanpath + 'BGU/' + str(i + 1).zfill(3) + '.mat'
#         scipy.io.savemat(savepath, Mat)
    
# for i in range(53):
#     matpath = str(i + 204).zfill(3) + '.mat'
#     matpath = BGU2 + 'BGU_HS_00' + matpath
#     Mat = {}
#     with h5py.File(matpath, 'r') as mat:
#         hyper = np.transpose(mat.get('rad'), [2,1,0])
#         hyper = (hyper-hyper.min())/(hyper.max()-hyper.min())
#         Mat['cube'] = hyper
#         rgbrecon = reconRGBfromNumpy(hyper)
#         Mat['rgb'] = rgbrecon
#         savepath = cleanpath + 'BGU/' + str(i + 204).zfill(3) + '.mat'
#         scipy.io.savemat(savepath, Mat)

for i in range(5):
    matpath = str(i * 2 + 257).zfill(3) + '.mat'
    matpath = BGU3 + 'BGU_HS_00' + matpath
    Mat = {}
    with h5py.File(matpath, 'r') as mat:
        hyper = np.transpose(mat.get('rad'), [2,1,0])
        hyper = (hyper-hyper.min())/(hyper.max()-hyper.min())
        Mat['cube'] = hyper
        rgbrecon = reconRGBfromNumpy(hyper)
        Mat['rgb'] = rgbrecon
        savepath = cleanpath + 'BGU/' + str(i + 257).zfill(3) + '.mat'
        scipy.io.savemat(savepath, Mat)
