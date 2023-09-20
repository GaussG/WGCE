import numpy as np
import tensorflow as tf
import scipy.io as sio
import os
import re
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import csv

def conv2d(input_, output_dim,
           k=5, s=2, stddev=0.02,
           name="conv2d",padding='SAME'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k, k, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, s, s, 1], padding=padding)
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        # 加偏置项，重塑维度
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv


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

def flame_cnn_prediction(Input,keep_prob=0.5,is_training=False,reuse=False):
    with tf.variable_scope('flame_cnn_prediction', reuse=reuse):
        with tf.variable_scope('CONV1'):
            net = conv2d(Input, 32, k=5, s=1, padding='SAME')
            net = tf.nn.relu(net)
        with tf.variable_scope('MAXPOOL1'):
            net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        with tf.variable_scope('CONV2'):
            net = conv2d(net, 64, k=5, s=1, padding='SAME')
            net = tf.nn.relu(net)
        with tf.variable_scope('MAXPOOL2'):
            net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
       # with tf.variable_scope('CONV3'):
           # net = conv2d(net, 128, k=3, s=1, padding='SAME')  # 17 x 17 x 256
            #net = tf.nn.relu(net)
       # with tf.variable_scope('MAXPOOL3'):
            #net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')  # 8 x 8 x 256
        with tf.variable_scope('FC1'):
            net = tf.layers.flatten(net)
            net = tf.nn.relu(linear(net, 100, 'g_h0_lin'))
            #net = tf.layers.dropout(net, keep_prob, training=is_training)
        with tf.variable_scope('output'):
            logits = linear(net, 1, 'logits')
        return logits


class flame_CNN(object):
    def __init__(self, sess, batch_size=10, test_batch_size=10,crop_height=100, crop_width=100,
                 c_dim=3,checkpoint_dir=None, keep_prob=0.5
                 ):
        self.sess = sess

        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.input_h = crop_height
        self.input_w = crop_width
        self.c_dim = c_dim
        self.checkpoint_dir = checkpoint_dir
        self.keep_prob = keep_prob

        self.build_model()

    def build_model(self):
        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size, self.input_h, self.input_w, self.c_dim], name="input_image")
        self.y = tf.placeholder(tf.float32, [self.batch_size, 1], name="label_y")
        self.inputs_test = tf.placeholder(
            tf.float32, [self.test_batch_size, self.input_h, self.input_w, self.c_dim], name="input_image_test")
        self.y_test = tf.placeholder(tf.float32, [self.test_batch_size, 1], name="label_y_test")
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.y_pre = flame_cnn_prediction(self.inputs,self.keep_prob,is_training=self.is_training)
        self.y_pre_test = flame_cnn_prediction(self.inputs_test, self.keep_prob, is_training=self.is_training,reuse=True)
        self.loss = tf.reduce_mean(tf.square(self.y_pre - self.y))
        self.loss_sum = tf.summary.scalar("loss", self.loss)
        self.init = tf.global_variables_initializer()

        self.saver = tf.train.Saver()

    def train(self,config):
        self.writer = tf.summary.FileWriter("./logs1", self.sess.graph)
        self.train_step = tf.train.AdamOptimizer(config.learning_rate,beta1=0.5).minimize(self.loss)
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        min_RMSE = 1
        counter = 0
        start_epoch = 0

        print(" [*] Reading checkpoint...")
        checkpoint_dir = os.path.join(self.checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            find_num = re.findall("(-?\d+\.?\d*e?-?\d*?)", ckpt_name)
            start_epoch = int(find_num[0]) + 1
            counter = int(find_num[1])
            min_RMSE = float(find_num[2])
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Succes to read {}".format(ckpt_name))

        # 读取训练集 测试集数据
        train_x = np.load('./data/train_x_new.npy')
        print(train_x.shape)
        train_y = np.load('./data/train_y_new.npy')
        test_x = np.load('./data/test_x_new.npy')
        test_y = np.load('./data/test_y_new.npy')
        train_x1 = np.load('./data/train_x_1_60.npy')
        train_y1 = np.load('./data/train_y_1_60.npy')
        gen_x = np.load('./gen_x_1.npy')
        print(gen_x.shape)
        gen_y = np.load('./gen_y_1.npy')
        train_ya = np.vstack([train_y1, gen_y])
        train_xa = np.vstack([train_x1, gen_x])
        train_y = np.vstack([train_ya, train_y[523:]])
        train_x = np.vstack([train_xa, train_x[523:]])
        np.random.seed(0)
        print(train_x.shape)
        idx_shuffle = np.arange(train_x.shape[0])
        np.random.shuffle(idx_shuffle)

        # 迭代阶段
        for epoch in range(start_epoch,config.epoch):
            for idx in range(0, train_x.shape[0]//self.batch_size):

                batch_train_x = train_x[idx_shuffle[idx * self.batch_size:(idx + 1) * self.batch_size]]
                bacth_train_y = train_y[idx_shuffle[idx * self.batch_size:(idx + 1) * self.batch_size]]
                _, summary_str = self.sess.run([self.train_step,self.loss_sum], feed_dict={self.inputs: batch_train_x,
                                                                                           self.y: bacth_train_y,
                                                                                           self.is_training: True})

                self.writer.add_summary(summary_str, counter)
                counter += 1

            test_y_pre = []
            for i in range(0, test_x.shape[0] // self.test_batch_size):
                batch_test_x = test_x[i * self.test_batch_size:(i + 1) * self.test_batch_size]
                batch_test_y_pre = self.sess.run(self.y_pre_test, feed_dict={self.inputs_test: batch_test_x,
                                                                        self.is_training: False})
                #batch_test_y_pre = np.around(batch_test_y_pre, 2)  # 将输出值保留到小数点后两位
                test_y_pre.append(batch_test_y_pre)

            test_y_pre = np.array(test_y_pre).reshape((-1, 1))
            test_RMSE = np.sqrt(np.mean(np.square(test_y_pre - test_y)))
            print('epoch {} test RMSE: {:.5f}'.format(epoch, test_RMSE))

            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

            if test_RMSE <= min_RMSE:
                min_RMSE = test_RMSE
                best_models = os.path.join(self.checkpoint_dir,
                                           'best_models_{}_{}_{:.5f}.ckpt'.format(epoch, counter, min_RMSE))
                print('------save:{}'.format(best_models))
                self.saver.save(self.sess, best_models)

        #best_models = os.path.join(self.checkpoint_dir, 'models_{}_{}_{:.4f}.ckpt'.format(epoch, counter, min_RMSE))
        #print('------save:{}'.format(best_models))
        #self.saver.save(self.sess, best_models)

    def test(self, config):
        self.sess.run(self.init)
        min_RMSE = 1
        counter = 0
        start_epoch = 0

        print(" [*] Reading checkpoint...")
        checkpoint_dir = os.path.join(self.checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            find_num = re.findall("(-?\d+\.?\d*e?-?\d*?)", ckpt_name)
            start_epoch = int(find_num[0]) + 1
            counter = int(find_num[1])
            min_RMSE = float(find_num[2])
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Succes to read {}".format(ckpt_name))

        # 读取训练集 测试集数据
        test_x = np.load('../../../../data/test_x_new.npy')
        test_y = np.load('../../../../data/test_y_new.npy')

        test_y_pre = []
        for i in range(0, test_x.shape[0] // self.test_batch_size):
            batch_test_x = test_x[i * self.test_batch_size:(i + 1) * self.test_batch_size]
            batch_test_y_pre = self.sess.run(self.y_pre_test, feed_dict={self.inputs_test: batch_test_x,
                                                                         self.is_training: False})
            # batch_test_y_pre = np.around(batch_test_y_pre, 2)  # 将输出值保留到小数点后两位
            test_y_pre.append(batch_test_y_pre)

        test_y_pre = np.array(test_y_pre).reshape((-1, 1))
        with open('vagan.npy','wb') as f:
            np.save(f,test_y_pre)
        erro = np.abs(test_y_pre-test_y)
        for i in range(test_y.shape[0]):
            print(erro[i])
        test_RMSE = np.sqrt(np.mean(np.square(test_y_pre-test_y)))
        # test_RE = np.sqrt(np.mean(np.square((test_y_pre-test_y)/test_y)))*100
        # test_MAE = np.max(np.abs(test_y_pre-test_y))

        mse = np.mean(np.square(test_y_pre - test_y))
        test_R2 = 1 - mse / np.var(test_y)  # 均方误差/方差

        print('RMSE:', test_RMSE)
        # print('RE:', test_RE,'%')
        # print('MAE:',test_MAE)
        print('R2:', test_R2)

        rows = np.array([[test_RMSE, test_R2]])
        rows = np.around(rows, 5)
        num = np.linspace(0, test_y.shape[0], test_y.shape[0])

        # plt.scatter(num, test_y_pre,s=1,c='r')
        # plt.plot(num, test_y, c='black')
        # plt.legend(['real data','CNN' ], loc='upper right')
        # plt.xlabel('Sample Number')
        # plt.ylabel('Oxygen Content')
        # if os.path.exists('./picture') is not True:
        #     os.makedirs('./picture')
        # plt.savefig(os.path.join('./picture', 'Fig.jpg'), format='jpg')
        # plt.show()

        print('___________________输出各区间的三个指标_____________________')
        header = ['RMSE', 'R2']

        with open('result_test.csv', 'w', newline='')as f:
            ff = csv.writer(f)
            ff.writerow(header)
            ff.writerows(rows)

            idx = np.array([0, 202, 438, 673, 911, 1046, 1218, 1453, 1698, 1933, 2153, 2368])
            for i in range(11):
                test_RMSE = np.sqrt(np.mean(np.square(test_y_pre[idx[i]:idx[i + 1]] - test_y[idx[i]:idx[i + 1]])))
                # test_RE = np.sqrt(np.mean(
                #     np.square((test_y_pre[idx[i]:idx[i + 1]] - test_y[idx[i]:idx[i + 1]]) / test_y[idx[i]:idx[i + 1]])))
                # test_MAE = np.max(np.abs(test_y_pre[idx[i]:idx[i + 1]] - test_y[idx[i]:idx[i + 1]]))
                mse = np.mean(np.square(test_y_pre[idx[i]:idx[i + 1]] - test_y[idx[i]:idx[i + 1]]))
                test_R2 = 1 - mse / np.var(test_y[idx[i]:idx[i + 1]])
                row = np.array([[test_RMSE, test_R2]])
                row = np.around(row, 5)  # 保留四位小数
                ff.writerows(row)






def show_all_variables():
    model_vars = tf.trainable_variables()  # 这个对象返回需要训练的变量列表
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)  # ？查不到，估计是显示变量什么


flags = tf.app.flags
flags.DEFINE_integer("epoch", 500, "Epoch to train [1400]")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate of for adam [0.001]")
flags.DEFINE_integer("batch_size", 85, "The size of batch images define by the number of train data")
flags.DEFINE_integer("test_batch_size", 64, "The size of batch images test for [64]")
flags.DEFINE_integer("input_height", 100, "The size of image to use (will be center cropped). [100]")
flags.DEFINE_integer("input_width", 100,
                     "The size of image to use (will be center cropped). If None, same value as input_height [100]")
flags.DEFINE_string("checkpoint_dir", "./checkpoints1", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_float("keep_prob", 0.5, "keep_prob [0.5]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

def main(_):

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    # 配置运行时的计算资源（GPU）
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        model = flame_CNN(
            sess,
            batch_size=FLAGS.batch_size,
            test_batch_size = FLAGS.test_batch_size,
            crop_height=FLAGS.input_height,
            crop_width=FLAGS.input_width,
            checkpoint_dir=FLAGS.checkpoint_dir,
            keep_prob = FLAGS.keep_prob
        )

        show_all_variables()

        if FLAGS.train:
            model.train(FLAGS)
        else:
            model.test(FLAGS)

if __name__ == '__main__':
    tf.app.run()


