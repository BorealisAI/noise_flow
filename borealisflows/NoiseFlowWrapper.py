# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import ast
import logging
import os

import tensorflow as tf

from borealisflows.noise_flow_model import NoiseFlow
from sidd.sidd_utils import restore_last_model

import numpy as np


class NoiseFlowWrapper:
    def __init__(self, path):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.nf_path = path
        self.nf_model = None
        self.sess = None
        self.saver = None

        self.x_shape = None
        self.x = None
        self.y = None
        self.nlf0 = None
        self.nlf1 = None
        self.iso = None
        self.cam = None
        self.is_training = None
        self.x_sample = None

        self.is_cond = True
        self.temp = 1.0

        self.hps = self.hps_loader(os.path.join(self.nf_path, 'hps.txt'))
        self.ckpt_dir = os.path.join(self.nf_path, 'ckpt')
        self.model_checkpoint_path = os.path.join(self.ckpt_dir, 'model.ckpt.best')
        self.load_noise_flow_model()

    def load_noise_flow_model(self):
        self.x_shape = [None, 32, 32, 4]
        if not hasattr(self.hps, 'x_shape'):
            setattr(self.hps, 'x_shape', self.x_shape)
        self.x = tf.placeholder(tf.float32, self.x_shape, name='noise_image')
        self.y = tf.placeholder(tf.float32, self.x_shape, name='clean_image')
        self.nlf0 = tf.placeholder(tf.float32, [None], name='nlf0')
        self.nlf1 = tf.placeholder(tf.float32, [None], name='nlf1')
        self.iso = tf.placeholder(tf.float32, [None], name='iso')
        self.cam = tf.placeholder(tf.float32, [None], name='cam')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        # graph1 = tf.Graph()
        # with graph1.as_default():

        self.logger.info('Building Noise Flow')
        self.nf_model = NoiseFlow(self.x_shape[1:], self.is_training, self.hps)

        self.logger.info('Creating sampling operation')
        self.x_sample = self.sample_sidd_tf()

        self.logger.info('Creating saver')
        self.saver = tf.train.Saver()

        self.logger.info('Creating session')
        self.sess = tf.Session()

        self.logger.info('Initializing variables')
        self.sess.run(tf.global_variables_initializer())

        self.logger.info('Restoring best model')
        # last_epoch = restore_last_model(self.ckpt_dir, self.sess, self.saver)
        self.saver.restore(self.sess, self.model_checkpoint_path)
        # import pdb
        # pdb.set_trace()

    def sample_noise_nf(self, batch_x, b1, b2, iso, cam):
        noise = None
        # sig = np.sqrt(b1 * batch_x + b2)  # in [0, 1]
        # noise = np.random.normal(0.0, sig, batch_x.shape)
        noise = self.sess.run(self.x_sample, feed_dict={self.y: batch_x, self.nlf0: [b1], self.nlf1: [b2],
                                                        self.iso: [iso], self.cam: [cam], self.is_training: True})
        return noise

    def sample_sidd_tf(self):
        if self.is_cond:
            x_sample = self.nf_model.sample(self.y, self.temp, self.y, self.nlf0, self.nlf1, self.iso, self.cam)
        else:
            x_sample = self.nf_model.sample(self.y, self.temp)
        return x_sample

    def hps_loader(self, path):
        import csv

        class Hps:
            pass

        hps = Hps()
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for pair in reader:
                if len(pair) < 2:
                    continue
                val = pair[1]
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except:
                        if val == 'True':
                            val = True
                        elif val == 'False':
                            val = False
                        # elif pair[0] == 'param_inits':
                            # import pdb
                            # pdb.set_trace()
                            # val = val.replace('\n', '')  # .replace('\r', '')
                            # val = ast.literal_eval(val)
                hps.__setattr__(pair[0], val)
        if hps.arch.__contains__('sdn5'):
            npcam = 3
        elif hps.arch.__contains__('sdn6'):
            npcam = 1
        # c_i = 1e-1
        c_i = 1.0
        beta1_i = -5.0 / c_i
        beta2_i = 0.0
        gain_params_i = np.ndarray([5])
        gain_params_i[:] = -5.0 / c_i
        cam_params_i = np.ndarray([npcam, 5])
        cam_params_i[:, :] = 1.0
        hps.param_inits = (c_i, beta1_i, beta2_i, gain_params_i, cam_params_i)
        return hps
