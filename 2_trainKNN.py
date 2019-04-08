# -*- coding: utf-8 -*
# 创建训练样本

import numpy as np
import cv2 as cv

#**************** 创建amount_kNN训练样本 ****************#

array = []
for i in range(1, 4):
    line = []
    for j in range(1, 16):
        k = 100*i+j
        strk = str(k)
        img = cv.imread('_amount\\A%s.jpg' % strk)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        line.append(gray)
    array.append(line)

x = np.array(array)
train = x[:,:].reshape(-1,800).astype(np.float32)

k = np.arange(1, 4)
train_labels = np.repeat(k,15)[:,np.newaxis]

np.savez('amount_data.npz',train=train, train_labels=train_labels)

#**************** 创建FCAC_kNN训练样本 ****************#

array = []
for i in range(1, 4):
    line = []
    for j in range(1, 16):
        k = 100*i+j
        strk = str(k)
        img = cv.imread('_FCAC\\F%s.jpg' % strk)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        line.append(gray)
    array.append(line)

x = np.array(array)
train = x[:,:].reshape(-1,3600).astype(np.float32)

k = np.arange(1, 4)
train_labels = np.repeat(k,15)[:,np.newaxis]

np.savez('FCAC_data.npz',train=train, train_labels=train_labels)
