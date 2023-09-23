import sys
sys.path.append('./')
from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import h5py
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
import torch
import torch.nn as nn
from options import opt
import scipy.io
import re
import matplotlib.pyplot as plt

root = '/work3/s212645/Spectral_Reconstruction/'
datanames = ['ICVL/', 'ARAD/', 'CAVE/']

def Rename(path):
    filelist = os.listdir(path)
    reg = re.compile(r'.*.mat')
    for file in filelist:
        if re.findall(reg, file):
            oldname = path + file
            number = file.split('.mat')[0]
            newnumber = number.zfill(3)
            newname = path + newnumber + '.mat'
            os.rename(oldname, newname)

def gen_test_list(matlist, startidx, size):
    testlist = []
    for i in range(size):
        testlist.append(matlist[i + startidx])
    return testlist

def cross_validation_lists(path, test_size):
    crossvaltrainlists = []
    crossvaltestlists = []
    filelist = os.listdir(path)
    filelist.sort()
    assert len(filelist) > test_size
    reg = re.compile(r'.*.mat')
    matlist = []
    for file in filelist:
        if re.match(reg, file):
            matlist.append(file)
    matlist.sort()
    all_length = len(matlist)
    assert all_length % test_size == 0
    for i in range(all_length // test_size):
        trainlist = []
        testlist = []
        testlist = gen_test_list(matlist, i * test_size, test_size)
        trainlist = matlist.copy()
        for mat in testlist:
            trainlist.remove(mat)
        crossvaltestlists.append(testlist)
        crossvaltrainlists.append(trainlist)
    return crossvaltrainlists, crossvaltestlists

def cross_validation_dataset(path, test_size, imsize):
    train_sets = []
    test_sets = []
    crossvaltrainlists, crossvaltestlists = cross_validation_lists(path, test_size)
    num_crossval = len(crossvaltrainlists)
    for i in range(num_crossval):
        train_sets.append(get_dataset(crossvaltrainlists[i], imsize))
        test_sets.append(get_dataset(crossvaltestlists[i], imsize))
    return train_sets, test_sets

def get_dataset(list, imsize):
    specs, rgbs = get_all_mats(list, imsize)
    return specs, rgbs

def split_train_test(path, test_size, imsize):
    filelist = os.listdir(path)
    filelist.sort()
    assert len(filelist) > test_size
    reg = re.compile(r'.*.mat')
    matlist = []
    for file in filelist:
        if re.match(reg, file):
            matlist.append(file)
    all_length = len(matlist)
    train_size = all_length - test_size
    matlist.sort()
    trainlist = []
    testlist = []
    for i in range(train_size):
        trainlist.append(matlist[i])
    for j in range(test_size):
        testlist.append(matlist[j + train_size])
    train_sets = get_dataset(trainlist, imsize)
    test_sets = get_dataset(testlist, imsize)
    return train_sets, test_sets

def get_all_mats(filelist, imsize):
    specs = []
    rgbs = []
    for file in filelist:
        mat = scipy.io.loadmat(path+file)
        spec, rgb = get_all_patches(mat, imsize)
        specs += spec
        rgbs += rgb
        # break
        newmat = {}
        newmat['cube'] = spec
        newmat['rgb'] = rgb
        scipy.io.savemat('/work3/s212645/ARAD/'+file, newmat)
        print('finish: ', file)
    return specs, rgbs
    
def Resize(hyper, rgb, h, w):
    hyper_s = cv2.resize(hyper, [h, w])
    rgb_s = cv2.resize(rgb, [h, w])
    return hyper_s, rgb_s

def data_resize(mat, imsize):
    spectral_images = []
    rgb_images = []
    hyper = np.float32(np.array(mat['cube']))
    rgb = np.float32(np.array(mat['rgb']))
    h = rgb.shape[0]
    w = rgb.shape[1]
    while min(h // imsize, w // imsize) >= 1:
        hyper_s, rgb_s = Resize(hyper, rgb, h, w)
        spectral_images.append(hyper_s)
        rgb_images.append(rgb_s)
        h = h // 2
        w = w // 2
    if h > imsize / 2 or w > imsize / 2 :
        hyper_s, rgb_s = Resize(hyper, rgb, imsize, imsize)
        spectral_images.append(hyper_s)
        rgb_images.append(rgb_s)
    return spectral_images, rgb_images

def patch_gen(array, imsize, h, w):
    return array[h-imsize:h, w-imsize:w, :]

def patch_image(array, imsize):
    h = array.shape[0]
    w = array.shape[1]
    assert h >= imsize
    assert w >= imsize
    iter_h = int(np.ceil(h / imsize) + 1)
    iter_w = int(np.ceil(w / imsize) + 1)
    patches = []
    for i in range(1, iter_h):
        for j in range(1, iter_w):
            patches.append(patch_gen(array, imsize, min(h, i * imsize), min(w, j * imsize)))
    return patches
    
def get_all_patches(mat, imsize):
    spectrals = []
    rgbs = []
    resize_spectrals, resize_rgbs = data_resize(mat, imsize)
    for i in range(len(resize_spectrals)):
        hyperpatches = patch_image(resize_spectrals[i], imsize)
        rgbpatches = patch_image(resize_rgbs[i], imsize)
        spectrals += hyperpatches
        rgbs += rgbpatches
    return spectrals, rgbs

if __name__ == '__main__':
    path = root + datanames[1]
    # spectral_images, rgb_images = get_all_mats(path, 128)
    # print(len(rgb_images))
    # print(rgb_images[-1].shape)
    train_sets, test_sets = split_train_test(path, 5, 128)
    print(len(train_sets[0]))
    print(sys.getsizeof(train_sets[0])/1024/1024)