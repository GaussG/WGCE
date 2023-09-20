# -*- coding: utf-8 -*-
"""
@author: Jiang
"""
import tensorflow as tf
from math import pi
EPSILON = tf.constant(1e-6, dtype=tf.float32)#创建常量
PI = tf.constant(pi, dtype=tf.float32)

#tensorboard 不同版本统一写法
try:    # 这些应该都是可视化用的变量或方法吧
  image_summary = tf.image_summary
  scalar_summary = tf.scalar_summary
  histogram_summary = tf.histogram_summary
  merge_summary = tf.merge_summary
  SummaryWriter = tf.train.SummaryWriter
except:
  image_summary = tf.summary.image
  scalar_summary = tf.summary.scalar
  histogram_summary = tf.summary.histogram
  merge_summary = tf.summary.merge
  SummaryWriter = tf.summary.FileWriter
  

if "concat_v2" in dir(tf):  #还提供了两个版本的concat函数，支持最新的concat_v2.
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat_v2(tensors, axis, *args, **kwargs)
else:   #concat,用于沿某个维度连接张量。
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)

def conv_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([
            x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)
    
def conv2d(input_, output_dim, 
          k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, 
          name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        #加偏置项，重塑维度
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())    
        return conv
 
    
class batch_norm(object):
    def __init__ (self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
         with tf.variable_scope(name):
             self.epsilon  = epsilon
             self.momentum = momentum
             self.name = name
    
    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

class layer_norm(object):
    def __init__ (self, eps = 1e-5,gamma=1,beta=0,name="layer_norm"):
         with tf.variable_scope(name):
             self.epsilon = eps
             self.gamma = gamma
             self.beta = beta
             self.name = name

    def __call__(self, x):
        self.x_mean = tf.reduce_mean(x, axis=(1, 2, 3), keepdims=True)
        self.x_var = tf.reduce_mean(tf.square(x - self.x_mean), axis=(1, 2, 3), keepdims=True)
        self.x_normalized = (x - self.x_mean) / tf.sqrt(self.x_var + self.epsilon)
        return self.gamma * self.x_normalized + self.beta


       
def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]], 
                                  initializer=tf.random_normal_initializer(stddev=stddev))
        
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])
        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, d_h, d_w, 1])
        
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        
        if with_w:
            return deconv, w, biases
        else:
            return deconv
        
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

        
#全连接操作(线性层)，返回节点输出值，w矩阵，每个节点的bias
def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()
    
    with tf.variable_scope(scope or "linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], 
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def GaussianSampleLayer(z_mu, z_lv, name='GaussianSampleLayer'):
    with tf.name_scope(name):
        eps = tf.random_normal(tf.shape(z_mu))
        std = tf.sqrt(tf.exp(z_lv))
        return tf.add(z_mu, tf.multiply(eps, std))

def GaussianKLD(mu1, lv1, mu2, lv2):
    ''' Kullback-Leibler divergence of two Gaussians
        *Assuming that each dimension is independent
        mu: mean
        lv: log variance
        Equation: http://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    #两个高斯分布的 Kullback-Leibler 散度
         *假设每个维度都是独立的
         mu：意思是
         lv：对数方差
    '''
    with tf.name_scope('GaussianKLD'):
        v1 = tf.exp(lv1)
        v2 = tf.exp(lv2)
        mu_diff_sq = tf.square(mu1 - mu2)#square 平方
        dimwise_kld = .5 * (
            (lv2 - lv1) + tf.div(v1 + mu_diff_sq, v2 + EPSILON) - 1.)#div 除法
        return tf.reduce_sum(dimwise_kld, -1)#按最后一个维度求和

def GaussianLogDensity(x, mu, log_var, name='GaussianLogDensity'):
    with tf.name_scope(name):
        c = tf.log(2. * PI)
        var = tf.exp(log_var)
        x_mu2 = tf.square(x - mu)   # [Issue] not sure the dim works or not?
        x_mu2_over_var = tf.div(x_mu2, var + EPSILON)
        log_prob = -0.5 * (c + log_var + x_mu2_over_var)
        log_prob = tf.reduce_sum(log_prob, -1)   # keep_dims=True,
        return log_prob