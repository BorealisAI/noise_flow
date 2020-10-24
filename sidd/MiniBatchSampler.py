# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import queue
import random
import time
from threading import Thread
import h5py
import numpy as np
from scipy.io import loadmat

from sidd.sidd_utils import pack_raw, get_nlf, load_one_tuple_images, sample_indices_uniform, \
    sidd_preprocess_standardize


class MiniBatchSampler:

    def __init__(self, patch_tuple_queue, minibatch_size=24, max_queue_size=16, n_threads=4, pat_stats=None):
        # queue to pull patch dictionaries from
        self.patch_tuple_queue = patch_tuple_queue
        self.mini_batch_size = minibatch_size
        self.pat_stats = pat_stats

        self.total_wait_time_get = 0
        self.total_wait_time_put = 0

        # initialize output queue
        self.max_queue_size = max_queue_size
        self.queue = queue.Queue(maxsize=self.max_queue_size)

        # initialize threads
        self.threads = []
        self.n_threads = n_threads
        for t in range(self.n_threads):
            self.threads.append(Thread(target=self.sample_minibatch_thread, args=(t, self.pat_stats)))
            self.threads[t].start()

    def sample_minibatch_thread(self, thread_id, pat_stats):
        mb_cnt = 0
        while True:
            # Dequeue patch dictionaries into mini-batch
            mini_batch_x = None
            mini_batch_y = None
            mini_batch_pid = None
            pat_dict = None
            for p in range(self.mini_batch_size):
                pat_dict = self.patch_tuple_queue.get()
                if p == 0:
                    p_shape = pat_dict['in'].shape
                    mini_batch_x = np.zeros((self.mini_batch_size, p_shape[1], p_shape[2], p_shape[3]))
                    mini_batch_y = np.zeros((self.mini_batch_size, p_shape[1], p_shape[2], p_shape[3]))
                    mini_batch_pid = np.zeros(self.mini_batch_size)
                mini_batch_x[p, :, :, :] = pat_dict['in']
                mini_batch_y[p, :, :, :] = pat_dict['gt']
                mini_batch_pid[p] = pat_dict['pid']  # patch index in image
            # only one value for the whole mini-batch:
            mini_batch_iso = [pat_dict['iso']]
            mini_batch_cam = [pat_dict['cam']]

            data = {'_x': mini_batch_x, '_y': mini_batch_y, 'pid': mini_batch_pid,
                    'iso': mini_batch_iso, 'cam': mini_batch_cam,
                    'fn': pat_dict['fn']}

            if 'nlf0' in pat_dict.keys(): # sRGB vs raw
                mini_batch_nlf0 = [pat_dict['nlf0']]
                mini_batch_nlf1 = [pat_dict['nlf1']]

                data.update({
                    'nlf0': mini_batch_nlf0,
                    'nlf1': mini_batch_nlf1,
                    'metadata': pat_dict['metadata']
                })

            self.queue.put(data)
            mb_cnt += 1
            # if self.patch_tuple_queue.empty():
            #     print('patch queue empty, # sampled minibatches = %d' % mb_cnt)

    def get_queue(self):
        return self.queue

    def get_total_wait_time(self):
        return self.total_wait_time_get, self.total_wait_time_put
