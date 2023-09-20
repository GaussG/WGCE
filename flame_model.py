# -*- coding: utf-8 -*-
"""
@author: Jiang
"""

from __future__ import division
import logging
import os
import math

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from glob import glob
import tensorflow as tf


from IMGops import *
import numpy as np
from NNops import *
import time
import tensorflow.contrib.layers as tcl
import tensorflow.contrib as tc
from tensorflow.contrib import slim

#计算卷积输出尺寸
def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

class flame_GAN(object):
    def __init__(self, sess, batch_size=10, crop_height=100, crop_width=100,
                 z_dim=100, c_dim=3, y_dim=1,gf_dim=64, df_dim=64,
                 dataset_name="flame", input_pattern="*.jpg",
                 checkpoint_dir=None, sample_dir=None
                 ):
        self.sess = sess
        
        self.batch_size = batch_size
        self.input_h = crop_height
        self.input_w = crop_width
        self.c_dim = c_dim
        
        self.z_dim = z_dim
        self.y_dim = y_dim
        #g和d网络的第一层卷积核数，可调，可不同，原DCGAN是64
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        
        #batch_norm
        self.d_bn0 = layer_norm(name='d_bn0')
        self.d_bn1 = layer_norm(name='d_bn1')
        self.d_bn2 = layer_norm(name='d_bn2')
        self.d_bn3 = layer_norm(name='d_bn3')

        self.e_bn0 = layer_norm(name='e_bn0')
        self.e_bn1 = layer_norm(name='e_bn1')
        self.e_bn2 = layer_norm(name='e_bn2')
        self.e_bn3 = layer_norm(name='e_bn3')
        
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2') 
        self.g_bn3 = batch_norm(name='g_bn3')
        
        
        self.dataset_name = dataset_name
        self.input_pattern = input_pattern
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir

        self.k = tf.constant(0.01)

        self.build_model()
        
    def build_model(self):

        self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name="label_y")
        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name="noise_z")
        #tensorboard
        self.z_sum = histogram_summary("z", self.z)
        self.inputs = tf.placeholder(
                tf.float32, [self.batch_size, self.input_h, self.input_w, self.c_dim], name="real_image")

        self.z_mu, self.z_lv = self.encoder(self.inputs, self.y,reuse=False)####
        self.z_all = GaussianSampleLayer(self.z_mu, self.z_lv)
        self.G_ = self.generator(self.z_all,self.y,reuse=False)

        self.G = self.generator(self.z,self.y,reuse=tf.AUTO_REUSE)

        self.D, self.D_logits = self.discriminator(self.inputs,self.y, reuse=False)

        #self.zh_mu, self.zh_lv = self.encoder(self.G, reuse=True)
        self.sampler = self.sampler(self.z,self.y,reuse=tf.AUTO_REUSE)  # 采样器
        self.D_E, self.D_logits_E = self.discriminator(self.G_,self.y, reuse=True)
        self.D_, self.D_logits_ = self.discriminator(self.G,self.y, reuse=True)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        ###loss
        self.d_loss = (tf.reduce_mean(self.D_logits_) - tf.reduce_mean(self.D_logits)) + (tf.reduce_mean(self.D_logits_E) - tf.reduce_mean(self.D_logits))
        # gradient penalty from WGAN-GP
        self.eps = tf.random_uniform([self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        self.x_penalty = self.eps * self.inputs + (1 - self.eps) * self.G
        self.penalty_logits = self.discriminator(self.x_penalty, self.y,reuse=True)
        self.gradients = tf.gradients(self.penalty_logits, self.x_penalty)[0]

        # 2范数
        self.grad_norm = tf.sqrt(tf.reduce_sum(tf.square(self.gradients), reduction_indices=[1, 2, 3]))
        self.grad_pen = tf.reduce_mean((self.grad_norm - 1.) ** 2)

        self.d_loss = self.d_loss + 10 * self.grad_pen

        #self.g_loss = -tf.reduce_mean(self.D_logits_E)
        self.Wasserstein_D = -self.d_loss
        #VAE的KL散度
        self.KL_loss = \
                tf.reduce_mean(
                    GaussianKLD(
                        self.z_mu, self.z_lv,
                        tf.zeros_like(self.z_mu), tf.zeros_like(self.z_lv)))#该操作返回与所有元素设置为零的tensor具有相同类型和形状的张量
        self.KL_loss /= 2.0

        ## VAE's Reconstruction Neg. Log-Likelihood (on the 'feature' space of Dx)VAE的重建否定。 对数似然（在 Dx 的“特征”空间上）
        self.Dis_loss = \
            tf.reduce_mean(
                GaussianLogDensity(
                    slim.flatten(self.inputs),
                    slim.flatten(self.G_),
                    tf.zeros_like(slim.flatten(self.inputs))))
        self.Dis_loss /= - 2.0

        # tensorboard,显示各种loss
        self.KL_loss_sum = scalar_summary('KL_loss', self.KL_loss)
        self.Dis_loss_sum = scalar_summary('Dis_loss', self.Dis_loss)

        self.vae_loss_sum = histogram_summary('z_', self.z_all)
        self.z_mu_sum = histogram_summary('z_mu', self.z_mu)
        self.z_lv_sum = histogram_summary('z_lv', self.z_lv)
        self.logit_sum = histogram_summary('logit_D', tf.concat([self.D_logits, self.D_logits_,self.D_logits_E], 0))#在第0维拼接

        #分别提取两个网络的训练变量
        t_vars = tf.trainable_variables()#打印要训练的变量
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        self.e_vars = [var for var in t_vars if 'encoder' in var.name]


        self.saver = tf.train.Saver(max_to_keep=20)#保存加载模型
    
    def train(self, config):
        self.global_step = tf.Variable(0, name='global_step')
        self.Wasserstein_D = self.Wasserstein_D / self.k

        self.obj_Dx = - self.Wasserstein_D * self.k
        self.obj_Gx = 20 * self.Wasserstein_D + self.Dis_loss
        self.obj_Ez = self.KL_loss + self.Dis_loss

        self.obj_Dx_sum = scalar_summary("loss_Dx", self.obj_Dx)
        self.obj_Gx_sum = scalar_summary("loss_Gx", self.obj_Gx)
        self.obj_Ez_sum = scalar_summary("loss_Ez", self.obj_Ez)

        d_optim = tf.train.AdamOptimizer(
            config.learning_rate, beta1=config.beta1).minimize(self.obj_Dx, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(
            config.learning_rate, beta1=config.beta1).minimize(self.obj_Gx, var_list=self.g_vars)
        e_optim = tf.train.AdamOptimizer(
            config.learning_rate, beta1=config.beta1).minimize(self.obj_Ez, var_list=self.e_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run() #兼容旧版本
        self.g_sum = merge_summary(
            [self.vae_loss_sum, self.z_sum, self.obj_Gx_sum])
        self.d_sum = merge_summary(
            [self.logit_sum,self.obj_Dx_sum])
        self.e_sum = merge_summary(
            [self.z_mu_sum, self.z_lv_sum, self.KL_loss_sum,self.Dis_loss_sum,self.obj_Ez_sum])

        self.writer = SummaryWriter("./logs", self.sess.graph)
        
        counter = 1
        #加载checkpoint
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load Succes")
        else:
            print(" [*] Load Failed")

        train_x = np.load('./data/train_x_1_60.npy')
        train_y = np.load('./data/train_y_1_60.npy')
        train_y = (train_y - np.min(train_y)) / (np.max(train_y) - np.min(train_y))
        np.random.seed(0)
        print(train_x.shape)

        idx_shuffle = np.arange(train_x.shape[0])
        np.random.shuffle(idx_shuffle)

        batch_idxs = train_x.shape[0] // self.batch_size #向下取整
                        
        #制造sample数据，训练时查看效果,均匀或正态随机，可尝试。取第0batch
        sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
        sample_imgs = train_x[idx_shuffle[0 * self.batch_size:(0 + 1) * self.batch_size]]
        sample_lbs = train_y[idx_shuffle[0 * self.batch_size:(0 + 1) * self.batch_size]]

        #迭代阶段
        start_time = time.time()              
        for epoch in range(config.epoch):
            for idx in range(0, batch_idxs):#0 1
                batch_images = train_x[idx_shuffle[idx * self.batch_size:(idx + 1) * self.batch_size]]
                batch_labels = train_y[idx_shuffle[idx * self.batch_size:(idx + 1) * self.batch_size]]

                #随机噪声，均匀/正态，可以尝试正态
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]) \
                                           .astype(np.float32)

                # updata D
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={
                                                   self.inputs: batch_images,
                                                   self.z: batch_z,
                                                   self.y: batch_labels
                                               })
                self.writer.add_summary(summary_str, counter)

                #updata G
                for _ in range(1):
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={
                                                       self.inputs: batch_images,
                                                       self.z: batch_z,
                                                       self.y: batch_labels
                                                   })
                    self.writer.add_summary(summary_str, counter)

                # updata E
                _, summary_str = self.sess.run([e_optim, self.e_sum],
                                               feed_dict={
                                                   self.inputs: batch_images,
                                                   self.z: batch_z,
                                                   self.y: batch_labels
                                               })
                self.writer.add_summary(summary_str, counter)


                errD = self.obj_Dx.eval({
                    self.inputs: batch_images, self.z: batch_z, self.y: batch_labels})
                errG = self.obj_Gx.eval({
                    self.inputs: batch_images, self.z: batch_z, self.y: batch_labels})
                errE = self.obj_Ez.eval({
                    self.inputs: batch_images, self.z: batch_z, self.y: batch_labels})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, e_loss: %.8f" \
                      % (epoch, idx, batch_idxs,
                         time.time() - start_time, errD, errG, errE ))
                
                if np.mod(counter, 500) == 1:
                    samples, d_loss, g_loss, e_loss = self.sess.run(
                            [self.sampler, self.obj_Dx, self.obj_Gx,self.obj_Ez],
                            feed_dict={
                                    self.z:sample_z,
                                    self.inputs:sample_imgs,
                                    self.y: sample_lbs
                                    })
                    save_images(samples, self.sample_dir, epoch, idx)
                    print("[Sample] d_loss: %.8f, g_loss: %.8f, e_loss: %.8f" % (d_loss, g_loss, e_loss))
                    
                if np.mod(counter, 500) == 1:#求余
                    self.save(config.checkpoint_dir, counter)
                    #每500次保存checkpoint,counter?
    def test(self):
        import matplotlib.pyplot as plt
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()  # 兼容旧版本

        counter = 1
        # 加载checkpoint
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load Succes")
        else:
            print(" [*] Load Failed")

        # 读取训练集 测试集数据
        test_x = np.load('./data/test_x.npy')
        test_y = np.load('./data/test_y.npy')
        sorted_indices = np.argsort(-test_y, axis=0)
        test_y = test_y[sorted_indices].reshape((-1, 1))
        test_x = test_x[sorted_indices].reshape((-1, 100, 100, 3))

        test_y_pre = []
        for i in range(0, test_x.shape[0] // self.batch_size):
            batch_test_x = test_x[i * self.batch_size:(i + 1) * self.batch_size]
            batch_test_y = test_y[i * self.batch_size:(i + 1) * self.batch_size]
            batch_test_y_pre = self.sess.run(self.D_logits, feed_dict={self.inputs: batch_test_x
                                                                    ,self.y: batch_test_y})
            # batch_test_y_pre = np.around(batch_test_y_pre,2) # 将输出值保留到小数点后两位
            test_y_pre.append(batch_test_y_pre)

        test_y_pre = np.array(test_y_pre).reshape((-1, 1))
        test_y = test_y[:(i + 1) * self.batch_size]
        test_RMSE = np.sqrt(np.mean(np.square(test_y_pre - test_y)))
        test_RE = np.sqrt(np.mean(np.square((test_y_pre - test_y) / test_y))) * 100
        test_MAE = np.mean(np.abs(test_y_pre - test_y))
        print('RMSE:', test_RMSE)
        print('RE:', test_RE, '%')
        print('MAE:', test_MAE)
        rows = np.array([[test_RMSE, test_RE, test_MAE]])
        rows = np.around(rows, 3)
        num = np.linspace(0, test_y.shape[0], test_y.shape[0])

        plt.scatter(num, test_y_pre, s=1, c='r')
        plt.plot(num, test_y, c='black')
        plt.legend(['real data', 'CNN'], loc='upper right')
        plt.xlabel('Sample Number')
        plt.ylabel('Oxygen Content')
        if os.path.exists('./picture') is not True:
            os.makedirs('./picture')
        plt.savefig(os.path.join('./picture', 'Fig_6333.jpg'), format='jpg')
        plt.show()

        print('___________________输出各区间的三个指标_____________________')
        header = ['RMSE', 'RE(%)', 'MAE']

        with open('result_6333.csv', 'w', newline='')as f:
            import csv
            ff = csv.writer(f)
            ff.writerow(header)
            ff.writerows(rows)

            idx = np.array([0, 212, 457, 703, 957, 1092, 1279, 1530, 1785, 2045, 2285, 2500])
            for i in range(11):
                test_RMSE = np.sqrt(np.mean(np.square(test_y_pre[idx[i]:idx[i + 1]] - test_y[idx[i]:idx[i + 1]])))
                test_RE = np.sqrt(np.mean(
                    np.square((test_y_pre[idx[i]:idx[i + 1]] - test_y[idx[i]:idx[i + 1]]) / test_y[idx[i]:idx[i + 1]])))
                test_MAE = np.mean(np.abs(test_y_pre[idx[i]:idx[i + 1]] - test_y[idx[i]:idx[i + 1]]))
                row = np.array([[test_RMSE, test_RE * 100, test_MAE]])
                row = np.around(row, 3)  # 保留三位小数
                ff.writerows(row)

    def encoder(self, image, y,reuse=False):
        with tf.variable_scope("encoder") as scope:
            if reuse:
                scope.reuse_variables()

            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])

            h0 = lrelu(self.e_bn0(conv2d(image, self.df_dim, name='e_h0_conv')))#100X100X32
            h0 = conv_concat(h0, yb)
            h1 = lrelu(self.e_bn1(conv2d(h0, self.df_dim * 2, name='e_h1_conv')))#50X50X64
            h1 = conv_concat(h1, yb)
            h2 = lrelu(self.e_bn2(conv2d(h1, self.df_dim * 4, name='e_h2_conv')))#25X25X128
            h2 = tf.reshape(h2, [self.batch_size, -1])
            h2 = tf.concat([h2, y], 1)

            self.z_mu = tcl.fully_connected(h2, 100, activation_fn=None)  # [batch_size,100]
            self.z_lv = tcl.fully_connected(h2, 100, activation_fn=None)

            return self.z_mu, self.z_lv

    def generator(self, z,y,reuse=False):
        with tf.variable_scope("generator") as scope:
            # 输出图片大小
            if reuse:
                scope.reuse_variables()
            s_h, s_w = self.input_h, self.input_w  # 120*160
            # 中间层卷积输出大小，从尾到头
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)  # 60*80
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)  # 30*40
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = concat([z, y], 1)
            z_ = linear(z, self.gf_dim * 2 * s_h4 * s_w4, 'g_h0_lin')

            h0 = tf.reshape(z_, [-1, s_h4, s_w4, self.gf_dim * 2])  # batch * 100 *100 *64
            h0 = tf.nn.relu(self.g_bn0(h0))
            h0 = conv_concat(h0, yb)

            h1 = deconv2d(h0, [self.batch_size, s_h2, s_w2, self.gf_dim], name='g_h1')  # batch * 50*50 *32
            h1 = tf.nn.relu(self.g_bn1(h1))
            h1 = conv_concat(h1, yb)

            h2 = deconv2d(h1, [self.batch_size, s_h, s_w, self.c_dim], name='g_h2')  # batch * 25 *25 *16

            # tf.nn.relu而不是lrelu, 避免负数值，因为要输出图片
            return tf.nn.sigmoid(h2)

    def discriminator(self, image,y, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])

            h0 = lrelu(self.d_bn0(conv2d(image, self.df_dim, name='d_h0_conv')))  # 100*100*32
            h0 = conv_concat(h0, yb)
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))  # 50*50*64
            h1 = conv_concat(h1, yb)
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))  # 25*25*128
            h2 = tf.reshape(h2, [self.batch_size, -1])
            h2 = tf.concat([h2, y], 1)
            h3 = linear(h2, 1, 'd_h3_lin')

            return tf.nn.sigmoid(h3), h3

    def sampler(self, z,y,reuse=False):
        with tf.variable_scope("generator") as scope:
            # 输出图片大小
            if reuse:
                scope.reuse_variables()
            # 输出图片大小
            s_h, s_w = self.input_h, self.input_w  # 120*160
            # 中间层卷积输出大小，从尾到头
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)  # 60*80
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)  # 30*40
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = concat([z, y], 1)
            z_ = linear(z, self.gf_dim * 2 * s_h4 * s_w4, 'g_h0_lin')

            h0 = tf.reshape(z_, [-1, s_h4, s_w4, self.gf_dim * 2])  # batch * 30 *40 *128
            h0 = tf.nn.relu(self.g_bn0(h0))
            h0 = conv_concat(h0, yb)

            h1 = deconv2d(h0, [self.batch_size, s_h2, s_w2, self.gf_dim], name='g_h1')  # batch * 60*80 *64
            h1 = tf.nn.relu(self.g_bn1(h1))
            h1 = conv_concat(h1, yb)

            h2 = deconv2d(h1, [self.batch_size, s_h, s_w, self.c_dim], name='g_h2')  # batch * 120 *160 *3
            # tf.nn.relu而不是lrelu, 避免负数值，因为要输出图片
            return tf.nn.sigmoid(h2)


    #装饰器将方法变成属性
    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
                self.dataset_name, self.batch_size, self.input_h, self.input_w)
    
    def save(self, checkpoint_dir, step):
        model_name = self.dataset_name
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        self.saver.save(self.sess, 
                        os.path.join(checkpoint_dir, model_name), 
                        global_step=step)
    
    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoint...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            ckpt_name ='flame-18001'
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Succes to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0            

