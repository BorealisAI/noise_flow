# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import queue
from threading import Thread
from time import sleep

from sidd.sidd_utils import pack_raw, get_nlf, load_one_tuple_images, load_one_tuple_srgb_images


class ImageLoader:

    def __init__(self, filename_tuple_queue, max_queue_size=4, n_threads=4, requeue=True):
        # queue to pull filename tuples from
        # make sure to re-queue filename tuples again (for future epochs)
        self.filename_tuple_queue = filename_tuple_queue
        self.temp_fns_queue = queue.Queue()

        self.total_wait_time_get = 0
        self.total_wait_time_put = 0

        # initialize output queue
        self.max_queue_size = max_queue_size
        self.queue = queue.Queue(maxsize=self.max_queue_size)

        # requeue items for more epochs
        self.requeue = requeue

        # initialize threads
        self.threads = []
        self.n_threads = n_threads
        for t in range(self.n_threads):
            self.threads.append(Thread(target=self.load_image_tuple_thread, args=[t]))
            self.threads[t].start()

    def load_image_tuple_thread(self, thread_id):
        while True:
            # Dequeue filename tuple and load image tuple
            filename_tuple = self.filename_tuple_queue.get()

            parts = str.split(filename_tuple[0], '/')
            fn = parts[-3] + '|' + parts[-1]

            if 'Srgb' in filename_tuple[0]:
              noise, gt, iso, cam = load_one_tuple_srgb_images(filename_tuple)
              im_dict = {'in': noise, 'gt': gt, 'iso': iso, 'cam': cam, 'fn': fn}
            else:
              noise, gt, var, nlf0, nlf1, iso, cam, metadata = load_one_tuple_images(filename_tuple)
              im_dict = {'in': noise, 'gt': gt, 'vr': var, 'nlf0': nlf0, 'nlf1': nlf1, 'iso': iso, 'cam': cam,
                        'fn': fn, 'metadata': metadata}

            # Enqueue image tuple
            self.queue.put(im_dict)

            # Enqueue filename tuple back, for more epochs
            if self.requeue:
                # keep file names in a temporary queue
                self.temp_fns_queue.put(filename_tuple)
                # if all file names are consumed, re-fill the queue again
                if self.filename_tuple_queue.empty():
                    # but wait till all images from last epoch are consumed
                    while not self.queue.empty():
                        sleep(1)
                    # then swap queues
                    tmp = self.filename_tuple_queue  # empty queue
                    self.filename_tuple_queue = self.temp_fns_queue  # the queue holding all file names
                    self.temp_fns_queue = tmp  # the empty queue

                self.filename_tuple_queue.put(filename_tuple)
            else:
                if self.filename_tuple_queue.empty():
                    break

    def get_queue(self):
        return self.queue

    def get_total_wait_time(self):
        return self.total_wait_time_get, self.total_wait_time_put
