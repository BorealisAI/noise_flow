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

from sidd.sidd_utils import pack_raw, get_nlf, load_one_tuple_images, sample_indices_uniform, sample_indices_random


class PatchSampler:

    def __init__(self, im_tuple_queue, patch_height=256, sampling='uniform', max_queue_size=256, n_threads=4,
                 n_reuse_image=0, n_pat_per_im=1, shuffle=True):
        # queue to pull image dictionaries from
        self.im_tuple_queue = im_tuple_queue
        self.patch_height = patch_height
        self.sampling = sampling
        self.n_reuse_image = n_reuse_image
        self.n_pat_per_im = n_pat_per_im
        self.shuffle = shuffle

        self.total_wait_time_get = 0
        self.total_wait_time_put = 0

        # initialize output queue
        self.max_queue_size = max_queue_size
        self.queue = queue.Queue(maxsize=self.max_queue_size)

        # initialize threads
        self.threads = []
        self.n_threads = n_threads
        for t in range(self.n_threads):
            self.threads.append(Thread(target=self.sample_patches_thread, args=(t, self.n_reuse_image)))
            self.threads[t].start()

    def sample_patches_thread(self, thread_id, n_reuse_image):
        cnt_im = 0
        pat_cnt = 0
        while True:
            # Dequeue image tuple and sample patches
            im_tuple = self.im_tuple_queue.get()
            cnt_im += 1
            H = im_tuple['in'].shape[1]
            W = im_tuple['in'].shape[2]
            if self.sampling == 'uniform':  # use all patches in image
                ii, jj, n_p = sample_indices_uniform(H, W, self.patch_height, self.patch_height, shuf=self.shuffle,
                                                     n_pat_per_im=self.n_pat_per_im)
                if n_p != self.n_pat_per_im:
                    print('# patches/image = %d != %d' % (n_p, self.n_pat_per_im))
                    print('fn = %s' % str(im_tuple['fn']))
                    import pdb
                    pdb.set_trace()
            else:  # use self.n_pat_per_im patches
                ii, jj = sample_indices_random(H, W, self.patch_height, self.patch_height, self.n_pat_per_im)
            pid = 0
            for (i, j) in zip(ii, jj):
                in_patch = im_tuple['in'][:, i:i + self.patch_height, j:j + self.patch_height, :]
                gt_patch = im_tuple['gt'][:, i:i + self.patch_height, j:j + self.patch_height, :]
                pat_dict = {'in': in_patch, 'gt': gt_patch, 'vr': [], 'nlf0': im_tuple['nlf0'],
                            'nlf1': im_tuple['nlf1'], 'iso': im_tuple['iso'], 'cam': im_tuple['cam'],
                            'fn': im_tuple['fn'], 'metadata': im_tuple['metadata'], 'pid': pid}
                pid += 1
                self.queue.put(pat_dict)
                pat_cnt += 1

            # if self.im_tuple_queue.empty():
            #     print('image queue is empty, # sampled patches = %d' % pat_cnt)

    def get_queue(self):
        return self.queue

    def get_total_wait_time(self):
        return self.total_wait_time_get, self.total_wait_time_put
