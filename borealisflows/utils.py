# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging

import matplotlib as mpl
import numpy as np
mpl.use('Agg')
import tensorflow as tf


def get_its(n_batch_train, n_batch_test, n_train, n_test):
    train_its = int(np.ceil(n_train / n_batch_train))
    test_its = int(np.ceil(n_test / n_batch_test))
    train_epoch = train_its * n_batch_train
    logging.info("Train epoch size: {}".format(train_epoch))
    return train_its, test_its


def int_shape(x):
    if str(x.get_shape()[0]) != '?':
        return list(map(int, x.get_shape()))
    return [-1] + list(map(int, x.get_shape()[1:]))


def squeeze2d(x, factor=2, squeeze_type='chessboard', x_shape=None):
    assert factor >= 1
    if factor == 1:
        return x
    if x_shape is None:
        shape = x.get_shape()
    else:
        shape = x_shape
    height = int(shape[1])
    width = int(shape[2])
    n_channels = int(shape[3])
    assert height % factor == 0 and width % factor == 0
    if squeeze_type == 'chessboard':
        # chess board
        x = tf.reshape(x, [-1, height // factor, factor,
                           width // factor, factor, n_channels])
        x = tf.transpose(x, [0, 1, 3, 5, 2, 4])
    elif squeeze_type == 'patch':
        # local patch
        x = tf.reshape(x, [-1, factor, height // factor,
                           factor, width // factor, n_channels])
        x = tf.transpose(x, [0, 2, 4, 5, 1, 3])
    else:
        print('Unknown squeeze type, using chessboard')
        # chess board
        x = tf.reshape(x, [-1, height // factor, factor,
                           width // factor, factor, n_channels])
        x = tf.transpose(x, [0, 1, 3, 5, 2, 4])
    x = tf.reshape(x, [-1, height // factor, width //
                       factor, n_channels * factor * factor])
    return x


def unsqueeze2d(x, factor=2, squeeze_type='chessboard'):
    assert factor >= 1
    if factor == 1:
        return x
    shape = x.get_shape()
    height = int(shape[1])
    width = int(shape[2])
    n_channels = int(shape[3])
    assert n_channels >= 4 and n_channels % 4 == 0
    x = tf.reshape(
        x, (-1, height, width, int(n_channels / factor ** 2), factor, factor))
    if squeeze_type == 'chessboard':
        # chess board
        x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
    elif squeeze_type == 'patch':
        # local path
        x = tf.transpose(x, [0, 4, 1, 5, 2, 3])
    else:
        print('Unknown squeeze type, using chessboard')
        # chess board
        x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
    x = tf.reshape(x, (-1, int(height * factor),
                       int(width * factor), int(n_channels / factor ** 2)))
    return x


# --- Logger ---
class ResultLogger(object):
    def __init__(self, path, columns, append=False):
        self.columns = columns
        mode = 'a' if append else 'w'
        self.f_log = open(path, mode)
        # self.f_log.write(json.dumps(kwargs) + '\n')
        if mode == 'w':
            self.f_log.write("\t".join(self.columns))

    def __del__(self):
        self.f_log.close()

    def log(self, run_info):
        run_strings = ["{0}".format(run_info[lc]) for lc in self.columns]
        self.f_log.write("\n")
        self.f_log.write("\t".join(run_strings))
        # self.f_log.write(json.dumps(kwargs) + '\n')
        self.f_log.flush()


def hps_logger(path, hps, layer_names, num_params):
    import csv
    # print(hps)
    with open(path, 'w') as f:
        w = csv.writer(f)
        for n in layer_names:
            w.writerow([n])
        w.writerow([num_params])
        for k, v in vars(hps).items():
            w.writerow([k, v])


def hps_loader(path):
    import csv

    class Hps():
        pass

    hps = Hps()
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for pair in reader:
            if len(pair) < 2:
                continue
            hps.__setattr__(pair[0], pair[1])
    return hps
