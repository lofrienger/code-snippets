from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# import json
# import argparse
# import random
# import random 
# import string
# non-standard dependencies:
# import h5py
# from six.moves import cPickle
import numpy as np
# import torch
# import torchvision.models as models
# import skimage.io

# from PIL import Image

# from torchvision import transforms as trn

# from pathlib import Path
  
import cv2

# disc_mask = prediction[prediction==0]
# print(disc_mask.shape)
# cup_mask = prediction[prediction==128]
# print(cup_mask.shape)

# Code reference: https://www.freesion.com/article/2264162053/
# 有关findContours()和drawContours()两个函数的详情可在上面的参考链接中查看。
# 可以在cv2.drawContours(image, [cnt], 0, (0, 255, 0), 1)这一行代码中修改函数参数，改动轮廓的颜色和大小。
def resize(mask):
    return cv2.resize(mask, (384, 384), interpolation=cv2.INTER_NEAREST)

def process_mask_1(mask_path):
    mask_2d = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    mask_2d = resize(mask_2d)
    # print(np.unique(mask_2d))
    # print('raw mask shape', mask_2d.shape) #raw mask shape (800, 800)
    # print(mask_2d)

    disc_mask = np.copy(mask_2d)
    cup_mask = np.copy(mask_2d)

    disc_mask[disc_mask==0] = 0
    disc_mask[disc_mask!=0] = 255
    cv2.imwrite('disc_mask_1.png', disc_mask)
    # print(disc_mask.shape)
    # print(disc_mask)

    cup_mask[cup_mask==128] = 0
    cup_mask[cup_mask!=0] = 255
    cv2.imwrite('cup_mask_1.png', cup_mask)
    # print(cup_mask.shape)
    # print(cup_mask)

def union_image_mask_disc_1(image_path, mask_path='disc_mask_1.png'):
    # 读取原图
    image = cv2.imread(image_path)
    image = resize(image)
    # print(image.shape) # (400, 500, 3)
    # print(image.size) # 600000
    # print(image.dtype) # uint8

    # 读取分割mask，这里本数据集中是白色背景黑色mask
    mask_2d = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_2d = resize(mask_2d)
    # 裁剪到和原图一样大小
    # mask_2d = mask_2d[0:400, 0:500]
    h, w = mask_2d.shape
    # cv2.imshow("2d", mask_2d)

    # 在OpenCV中，查找轮廓是从黑色背景中查找白色对象，所以要转成黑色背景白色mask
    mask_3d = np.ones((h, w), dtype='uint8')*255
    # mask_3d_color = np.zeros((h,w,3),dtype='uint8')
    mask_3d[mask_2d[:, :] == 255] = 0
    # cv2.imshow("3d", mask_3d)


    ret, thresh = cv2.threshold(mask_3d, 127, 255, 0)
    # old cv2 version
    # im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # new cv2 version
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    cv2.drawContours(image, [cnt], 0, (0, 0, 255), 2) #red
    # 打开画了轮廓之后的图像
    # cv2.imshow('mask', image)
    # k = cv2.waitKey(0)
    # if k == 27:
    #     cv2.destroyAllWindows()
    # 保存图像
    # cv2.imwrite("./image/result/" + str(num) + ".bmp", image)
    cv2.imwrite('1.png', image)
    
def union_image_mask_cup_1(image_path='1.png', mask_path='cup_mask_1.png'):
    # 读取原图
    image = cv2.imread(image_path)
    image = resize(image)
    # print(image.shape) # (400, 500, 3)
    # print(image.size) # 600000
    # print(image.dtype) # uint8

    # 读取分割mask，这里本数据集中是白色背景黑色mask
    mask_2d = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_2d = resize(mask_2d)
    # 裁剪到和原图一样大小
    # mask_2d = mask_2d[0:400, 0:500]
    h, w = mask_2d.shape
    # cv2.imshow("2d", mask_2d)

    # 在OpenCV中，查找轮廓是从黑色背景中查找白色对象，所以要转成黑色背景白色mask
    mask_3d = np.ones((h, w), dtype='uint8')*255
    # mask_3d_color = np.zeros((h,w,3),dtype='uint8')
    mask_3d[mask_2d[:, :] == 255] = 0
    # cv2.imshow("3d", mask_3d)


    ret, thresh = cv2.threshold(mask_3d, 127, 255, 0)
    # old cv2 version
    # im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # new cv2 version
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    cv2.drawContours(image, [cnt], 0, (0, 0, 255), 2) # red
    # 打开画了轮廓之后的图像
    # cv2.imshow('mask', image)
    # k = cv2.waitKey(0)
    # if k == 27:
    #     cv2.destroyAllWindows()
    # 保存图像
    # cv2.imwrite("./image/result/" + str(num) + ".bmp", image)
    cv2.imwrite('2.png', image)

def process_mask_2(mask_path):
    mask_2d = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_2d = resize(mask_2d)
    # print('raw mask shape', mask_2d.shape) #raw mask shape (800, 800)
    # print(mask_2d)

    disc_mask = np.copy(mask_2d)
    cup_mask = np.copy(mask_2d)

    disc_mask[disc_mask==0] = 0
    disc_mask[disc_mask!=0] = 255
    cv2.imwrite("disc_mask_2.png", disc_mask)
    # print(disc_mask.shape)
    # print(disc_mask)

    cup_mask[cup_mask==128] = 0
    cup_mask[cup_mask!=0] = 255
    cv2.imwrite("cup_mask_2.png", cup_mask)

def union_image_mask_disc_2(image_path='2.png', mask_path='disc_mask_2.png'):
    # 读取原图
    image = cv2.imread(image_path)
    image = resize(image)
    # print(image.shape) # (400, 500, 3)
    # print(image.size) # 600000
    # print(image.dtype) # uint8

    # 读取分割mask，这里本数据集中是白色背景黑色mask
    mask_2d = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_2d = resize(mask_2d)
    # 裁剪到和原图一样大小
    # mask_2d = mask_2d[0:400, 0:500]
    h, w = mask_2d.shape
    # cv2.imshow("2d", mask_2d)

    # 在OpenCV中，查找轮廓是从黑色背景中查找白色对象，所以要转成黑色背景白色mask
    mask_3d = np.ones((h, w), dtype='uint8')*255
    # mask_3d_color = np.zeros((h,w,3),dtype='uint8')
    mask_3d[mask_2d[:, :] == 255] = 0
    # cv2.imshow("3d", mask_3d)


    ret, thresh = cv2.threshold(mask_3d, 127, 255, 0)
    # old cv2 version
    # im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # new cv2 version
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    if contours:
        cnt = contours[0]
        cv2.drawContours(image, [cnt], 0, (255, 0, 0), 2) # blue
    # 打开画了轮廓之后的图像
    # cv2.imshow('mask', image)
    # k = cv2.waitKey(0)
    # if k == 27:
    #     cv2.destroyAllWindows()
    # 保存图像
    # cv2.imwrite("./image/result/" + str(num) + ".bmp", image)
    cv2.imwrite("3.png", image)

def union_image_mask_cup_2(output, image_path='3.png', mask_path='cup_mask_2.png'):
    # 读取原图
    image = cv2.imread(image_path)
    image = resize(image)
    # print(image.shape) # (400, 500, 3)
    # print(image.size) # 600000
    # print(image.dtype) # uint8

    # 读取分割mask，这里本数据集中是白色背景黑色mask
    mask_2d = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_2d = resize(mask_2d)
    # 裁剪到和原图一样大小
    # mask_2d = mask_2d[0:400, 0:500]
    h, w = mask_2d.shape
    # cv2.imshow("2d", mask_2d)

    # 在OpenCV中，查找轮廓是从黑色背景中查找白色对象，所以要转成黑色背景白色mask
    mask_3d = np.ones((h, w), dtype='uint8')*255
    # mask_3d_color = np.zeros((h,w,3),dtype='uint8')
    mask_3d[mask_2d[:, :] == 255] = 0
    # cv2.imshow("3d", mask_3d)


    ret, thresh = cv2.threshold(mask_3d, 127, 255, 0)
    # old cv2 version
    # im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # new cv2 version
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours: # has contours
        cnt = contours[0]
        cv2.drawContours(image, [cnt], 0, (0, 255, 0), 2)# green
    # 打开画了轮廓之后的图像
    # cv2.imshow('mask', image)
    # k = cv2.waitKey(0)
    # if k == 27:
    #     cv2.destroyAllWindows()
    # 保存图像
    # cv2.imwrite("./image/result/" + str(num) + ".bmp", image)
    cv2.imwrite(output, image)

image_path = '/home/ren2/data2/wa_dataset/Fundus_ROIs/Domain1/test/image/gdrishtiGS_067.png'
gt_mask_path = '/home/ren2/data2/wa_dataset/Fundus_ROIs/Domain1/test/mask/gdrishtiGS_067.png'
pr_mask_path = '/home/ren2/data2/wa/FDA_Seg/masks/pred/Domain1/CFAM/gdrishtiGS_067.png'

root_path = '/home/ren2/data2/wa_dataset/Fundus_ROIs/'
pre_root = '/home/ren2/data2/wa/FDA_Seg/masks/pred/'
# output_root = '/home/ren2/data2/wa/FDA_Seg/masks/overlay/class_diff'
output_root = '/home/ren2/data2/wa/FDA_Seg/masks/overlay/temp'

domain_list = ['Domain1', 'Domain2', 'Domain3', 'Domain4']
method_list = ['UNet', 'FDA', 'FACT', 'CFDA', 'CFAM']

# selected_image_list = ['gdrishtiGS_027.png', 'S-31-L.png', 'n0291.png', 'V0128.png']
selected_image_list = ['gdrishtiGS_027.png','S-31-L.png', 'n0291.png', 'V0128.png']

for i, domain in enumerate(domain_list):
    image_list = os.listdir(os.path.join(root_path, domain, 'test', 'image'))
    print(domain, len(image_list))
    for method in method_list:
        print(method)
        for image in image_list:
            # print(image)
            if image == selected_image_list[i]:
                gt_mask = os.path.join(root_path, domain, 'test', 'mask', image)
                image_path = os.path.join(root_path, domain, 'test', 'image', image)
                pr_mask = os.path.join(pre_root, domain, method, image)
                output_file = os.path.join(output_root, image.split('.')[0]+'-'+method+'.png')

                process_mask_1(gt_mask)
                union_image_mask_disc_1(image_path)
                union_image_mask_cup_1()
                # import sys
                # sys.exit(0)
                # process_mask_2(pr_mask)
                # union_image_mask_disc_2()
                # union_image_mask_cup_2(output_file)

# 原文链接：https://blog.csdn.net/miao0967020148/article/details/88623631
# OpenCV旧版，返回三个参数：
# im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 要想返回三个参数：
# 把OpenCV 降级成3.4.3.18 就可以了，在终端输入pip install opencv-python==3.4.3.18

# OpenCV 新版调用，返回两个参数：
#  contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


