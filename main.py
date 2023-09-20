# -*- coding: utf-8 -*-
"""
@author: Jiang
"""

import os
import numpy as np
from pearl_model import flood_GAN
import pprint #pprint模块提供了打印出任何python数据结构的类和方法
from IMGops import *

import tensorflow as tf

import tensorflow.contrib.slim as slim #TF-slim库

def show_all_variables():
    model_vars = tf.trainable_variables() #这个对象返回需要训练的变量列表
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)#？查不到，估计是显示变量什么

flags = tf.app.flags
flags.DEFINE_integer("epoch", 6000, "Epoch to train [1400]")   # 400 epoch
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "`Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", 20000000, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 20, "The size of batch images [16]")
flags.DEFINE_integer("input_height", 100, "The size of image to use (will be center cropped). [250]")
flags.DEFINE_integer("input_width", 100, "The size of image to use (will be center cropped). If None, same value as input_height [250]")
flags.DEFINE_string("input_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "./checkpoints", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "./samples_500", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train",True, "True for training, False for testing [False]")
flags.DEFINE_boolean("test", False, "True for testing,")
FLAGS = flags.FLAGS

def main(_):

    pp = pprint.PrettyPrinter() #创建一个PrettyPrinter对象，用于打印输出数据
    pp.pprint(flags.FLAGS.__flags)    #打印出Flags
    
    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
        
    #配置运行时的计算资源（GPU）
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True
    #tf.reset_default_graph()
    with tf.Session(config=run_config) as sess:

        pearl_gan = flood_GAN(
                sess,
                batch_size = FLAGS.batch_size,
                crop_height = FLAGS.input_height,
                crop_width = FLAGS.input_width,
                input_pattern = FLAGS.input_pattern,
                checkpoint_dir = FLAGS.checkpoint_dir,
                sample_dir = FLAGS.sample_dir
                )
        
        show_all_variables()
        
        if FLAGS.train:
            pearl_gan.train(FLAGS)
        elif FLAGS.test:
            pearl_gan.test()
        else:
            if not pearl_gan.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test mode")
            visualize(sess, pearl_gan, FLAGS)

    


if __name__ == '__main__':
    tf.app.run()
    
    
    
    
    
    