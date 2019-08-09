# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import tensorflow as tf
import numpy as np

from borealisflows.utils import squeeze2d

tfd = tf.contrib.distributions
tfb = tfd.bijectors

# Settings
DTYPE = tf.float32
NP_DTYPE = np.float32


class AffineCouplingCondXYG(tfb.Bijector):

    def __init__(
            self,
            x_shape,
            shift_and_log_scale_fn,
            layer_id=0,
            last_layer=False,
            validate_args=False,
            name="real_nvp"):
        super(AffineCouplingCondXYG, self).__init__(
            forward_min_event_ndims=1,
            is_constant_jacobian=False,
            validate_args=validate_args,
            name=name)
        self.x_shape = x_shape
        self.i0, self.i1, self.ic = x_shape
        self._last_layer = last_layer
        self.id = layer_id
        self._shift_and_log_scale_fn = shift_and_log_scale_fn
        self.scale = tf.get_variable(
            "rescaling_scale{}".format(self.id), [], dtype=DTYPE,
            initializer=tf.constant_initializer(1e-4))  # initializer=tf.zeros_initializer())

    def _forward(self, x, yy, nlf0=None, nlf1=None, iso=None, cam=None):
        # print('_forward-------')
        # import pdb
        # pdb.set_trace()
        if self._last_layer:
            x = tf.reshape(x, (-1, self.i0, self.i1, self.ic))
            yy = tf.reshape(yy, (-1, self.i0, self.i1, self.ic))
        if 2 * x.shape[1] == yy.shape[1]:
            yy = squeeze2d(yy, 2)
        x0 = x[:, :, :, :self.ic // 2]
        x1 = x[:, :, :, self.ic // 2:]
        x0yy = tf.concat([x0, yy], axis=-1)
        shift, log_scale = self._shift_and_log_scale_fn(x0yy, iso)
        log_scale = self.scale * tf.tanh(log_scale)
        y1 = x1
        if shift is not None:
            y1 -= shift
        if log_scale is not None:
            y1 *= tf.exp(-log_scale)
        y = tf.concat([x0, y1], axis=-1)
        return y

    def _inverse(self, y, yy, nlf0=None, nlf1=None, iso=None, cam=None):
        y0 = y[:, :, :, :self.ic // 2]
        y1 = y[:, :, :, self.ic // 2:]
        y0yy = tf.concat([y0, yy], axis=-1)
        shift, log_scale = self._shift_and_log_scale_fn(y0yy, iso)
        tf.summary.histogram('cXY shift', shift)
        tf.summary.histogram('cXY log_scale', log_scale)
        log_scale = self.scale * tf.tanh(log_scale)
        x1 = y1
        if log_scale is not None:
            x1 *= tf.exp(log_scale)
        if shift is not None:
            x1 += shift
        x = tf.concat([y0, x1], axis=-1)
        if self._last_layer:
            return tf.layers.flatten(x)
        return x

    def _forward_log_det_jacobian(self, x, yy, nlf0=None, nlf1=None, iso=None, cam=None):
        if self._last_layer:
            x = tf.reshape(x, (-1, self.i0, self.i1, self.ic))
            yy = tf.reshape(yy, (-1, self.i0, self.i1, self.ic))
        if 2 * x.shape[1] == yy.shape[1]:
            yy = squeeze2d(yy, 2)
        x0 = x[:, :, :, :self.ic // 2]
        x0yy = tf.concat([x0, yy], axis=-1)
        _, log_scale = self._shift_and_log_scale_fn(x0yy, iso)
        log_scale = self.scale * tf.tanh(log_scale)
        if log_scale is None:
            return tf.constant(0., dtype=x.dtype, name="fldj")
        return -tf.reduce_sum(log_scale, axis=[1, 2, 3])

    def _inverse_log_det_jacobian(self, z, yy, nlf0=None, nlf1=None, iso=None, cam=None):
        z0 = z[:, :, :, :self.ic // 2]
        z0yy = tf.concat([z0, yy], axis=-1)
        _, log_scale = self._shift_and_log_scale_fn(z0yy, iso)
        tf.summary.histogram('cXY log_scale', log_scale)
        log_scale = self.scale * tf.tanh(log_scale)
        if log_scale is None:
            return tf.constant(0., dtype=z.dtype, name="ildj")
        return tf.reduce_sum(log_scale, axis=[1, 2, 3])

    def _forward_and_log_det_jacobian(self, x, yy, nlf0=None, nlf1=None, iso=None, cam=None):
        if self._last_layer:
            x = tf.reshape(x, (-1, self.i0, self.i1, self.ic))
            yy = tf.reshape(yy, (-1, self.i0, self.i1, self.ic))
        if 2 * x.shape[1] == yy.shape[1]:
            yy = squeeze2d(yy, 2)
        x0 = x[:, :, :, :self.ic // 2]
        x1 = x[:, :, :, self.ic // 2:]
        x0yy = tf.concat([x0, yy], axis=-1)
        shift, log_scale = self._shift_and_log_scale_fn(x0yy, iso)
        log_scale = self.scale * tf.tanh(log_scale)
        y1 = x1
        if shift is not None:
            y1 -= shift
        if log_scale is not None:
            y1 *= tf.exp(-log_scale)
        y = tf.concat([x0, y1], axis=-1)
        if log_scale is None:
            log_abs_det_J = tf.constant(0., dtype=x.dtype, name="fldj")
        else:
            log_abs_det_J = -tf.reduce_sum(log_scale, axis=[1, 2, 3])
        return y, log_abs_det_J

    def _inverse_and_log_det_jacobian(self, y, yy, nlf0=None, nlf1=None, iso=None, cam=None):
        # Performs scale and shift.
        y0 = y[:, :, :, :self.ic // 2]
        y1 = y[:, :, :, self.ic // 2:]
        y0yy = tf.concat([y0, yy], axis=-1)
        shift, log_scale = self._shift_and_log_scale_fn(y0yy, iso)
        tf.summary.histogram('cXY shift', shift)
        tf.summary.histogram('cXY log_scale', log_scale)
        log_scale = self.scale * tf.tanh(log_scale)
        x1 = y1
        if log_scale is not None:
            x1 *= tf.exp(log_scale)
        if shift is not None:
            x1 += shift
        x = tf.concat([y0, x1], axis=-1)
        if log_scale is None:
            log_abs_det_J_inv = tf.constant(0., dtype=y.dtype, name="ildj")
        else:
            log_abs_det_J_inv = tf.reduce_sum(log_scale, axis=[1, 2, 3])
        if self._last_layer:
            return tf.layers.flatten(x), log_abs_det_J_inv
        return x, log_abs_det_J_inv

