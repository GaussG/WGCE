# -*- coding: utf-8 -*-
"""
Created on Mon Sep 04 20:07:54 2017

@author: Zz-Chen
"""
from __future__ import division
import os
import scipy.misc
import numpy as np
from glob import glob
import imageio
def center_crop(img, crop_h, crop_w):
    if crop_w is None:
        crop_w = crop_h
    h,w = img.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return img[j:j+crop_h, i:i+crop_w]


#载入单颗珍珠数据和标签，具体操作：center crop, 五视图通道叠合, 标签onehot
#return 单珍珠五视图，onehot标签
def load_one_label(y_dim,data_list):
    label = data_list.split('\\')[-1]
    label = label.split('_')[0]
    label = int(label)
    label_vec = np.zeros((y_dim), dtype=np.float)
    label_vec[label] = 1.0
    return label_vec

def load_one_flame(start_number):
    data_list = glob("./data/train/*.jpg")
    img = scipy.misc.imread(data_list[start_number]).astype(np.float)
    #归一化，使参数同量级，返回
    return img/255.

def save_images(images, save_dir, epoch, idx):
    for j in range(images.shape[0]):
        img = images[j]
        save_path = os.path.join(save_dir,
        'tr_ep{}_b{}_{}.jpg'.format(epoch, idx, j))
        img1 = np.uint8(img * 255)
        imageio.imsave(save_path, img1)


def test_save_images(images, save_dir,predict_num, loop):
    for j in range(images.shape[0]):
        img = images[j]
        #save_dir_i = os.path.join(save_dir, 'train_{}'.format(class_num))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir,
                                 '{}_{}.jpg'.format(predict_num[j][0], loop + j))
        img1 = np.uint8(img * 255)
        imageio.imsave(save_path, img1)


def visualize(sess, pearl_gan, config):
    sample_x = np.zeros([1, 100, 100, 3])
    sample_y = np.zeros([1, 1])
    np.random.seed(0)
    train_y = np.load('./data/train_y_1_60.npy')
    sort_idx = np.argsort(train_y, axis=0)
    train_y = train_y[sort_idx].reshape((-1, 1))
    sample_y_unique = np.unique(train_y).reshape((-1, 1))
    count = np.zeros([sample_y_unique.shape[0], 1])

    for i in range(train_y.shape[0]):
        for j in range(sample_y_unique.shape[0]):
            if train_y[i] == sample_y_unique[j][0]:
                count[j] += 1
                break
    count_use = count / np.sum(count) * 470
    count_use = np.around(count_use).astype(np.int)  # 每个标签取多少个样本

    # print(count_use)
    print(np.sum(count_use))
    iter_idx = np.ceil(count_use / 10).astype(np.int)

    # print(iter_idx)
    for i in range(iter_idx.shape[0]):
        sample_x_i = np.zeros([1, 100, 100, 3])
        sample_y_i = np.zeros([1, 1])
        for j in range(iter_idx[i][0]):
            test_z = np.random.uniform(-0.99, 0.99, size=(config.batch_size, pearl_gan.z_dim))
            test_y = (sample_y_unique[i][0] - np.min(sample_y_unique)) / (
                    np.max(sample_y_unique) - np.min(sample_y_unique))
            test_y = np.array([[test_y]] * 20)

            test_sample = sess.run(pearl_gan.sampler,
                                   feed_dict={
                                       pearl_gan.z: test_z,
                                       pearl_gan.y: test_y
                                   })
            test_y = test_y * (np.max(sample_y_unique) - np.min(sample_y_unique)) + np.min(sample_y_unique)
            test_save_images(test_sample, config.sample_dir, test_y, config.batch_size * j * i)
            sample_x_i = np.vstack([sample_x_i, test_sample])
            sample_y_i = np.vstack([sample_y_i, test_y])
        sample_x_i = sample_x_i[1:count_use[i][0] + 1]
        sample_y_i = sample_y_i[1:count_use[i][0] + 1]

        sample_x = np.vstack([sample_x, sample_x_i])
        sample_y = np.vstack([sample_y, sample_y_i])
    sample_x = sample_x[1:]
    sample_y = sample_y[1:]

    print(sample_x.shape)
    print(sample_y.shape)
    f = open('gen_x_1.npy', 'wb')
    np.save(f, sample_x)
    f.close()
    f = open('gen_y_1.npy', 'wb')
    np.save(f, sample_y)
    f.close()

