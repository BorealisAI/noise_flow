# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import tensorflow as tf
import numpy as np

from borealisflows.noise_flow_layers.cond_utils import sdn_model_params_ex3, sdn_model_params_ex4, sdn_model_params_ex5
from borealisflows.utils import squeeze2d

tfd = tf.contrib.distributions
tfb = tfd.bijectors

# Settings
DTYPE = tf.float32
NP_DTYPE = np.float32


class AffineCouplingSdnEx5(tfb.Bijector):

    def __init__(
            self,
            x_shape,
            shift_and_log_scale_fn,
            layer_id=0,
            last_layer=False,
            validate_args=False,
            name="real_nvp",
            gain_init=0.0,
            param_inits=False):
        super(AffineCouplingSdnEx5, self).__init__(
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
        self.gain_init = gain_init
        self.param_inits = param_inits

    def _forward(self, x, yy, nlf0=None, nlf1=None, iso=None, cam=None):
        if self._last_layer:
            x = tf.reshape(x, (-1, self.i0, self.i1, self.ic))
            yy = tf.reshape(yy, (-1, self.i0, self.i1, self.ic))

        if yy.shape[1] == 2 * x.shape[1]:  # needs squeezing
            yy = squeeze2d(yy, 2)

        log_scale = sdn_model_params_ex5(yy, iso, self.gain_init, cam, self.param_inits)
        log_scale = tf.tanh(log_scale)
        shift = 0.0

        y = x
        if log_scale is not None:
            y *= tf.exp(log_scale)
        if shift is not None:
            y += shift
        return y

    def _inverse(self, y, yy, nlf0=None, nlf1=None, iso=None, cam=None):
        log_scale = sdn_model_params_ex5(yy, iso, self.gain_init, cam, self.param_inits)
        log_scale = tf.tanh(log_scale)

        # tf.summary.histogram('fitSDN_beta1', beta1)
        # tf.summary.histogram('fitSDN_beta2', beta2)

        x = y
        if log_scale is not None:
            x /= tf.exp(log_scale)
        if self._last_layer:
            return tf.layers.flatten(x)
        return x

    def _forward_log_det_jacobian(self, x, yy, nlf0=None, nlf1=None, iso=None, cam=None):
        if self._last_layer:
            x = tf.reshape(x, (-1, self.i0, self.i1, self.ic))
            yy = tf.reshape(yy, (-1, self.i0, self.i1, self.ic))

        log_scale = sdn_model_params_ex5(yy, iso, self.gain_init, cam, self.param_inits)
        log_scale = tf.tanh(log_scale)

        if log_scale is None:
            return tf.constant(0., dtype=x.dtype, name="fldj")
        return tf.reduce_sum(log_scale, axis=[1, 2, 3])

    def _inverse_log_det_jacobian(self, z, yy, nlf0=None, nlf1=None, iso=None, cam=None):
        log_scale = sdn_model_params_ex5(yy, iso, self.gain_init, cam, self.param_inits)
        log_scale = tf.tanh(log_scale)

        # tf.summary.histogram('fitSDN_beta1', beta1)
        # tf.summary.histogram('fitSDN_beta2', beta2)

        if log_scale is None:
            return tf.constant(0., dtype=z.dtype, name="ildj")
        return - tf.reduce_sum(log_scale, axis=[1, 2, 3])

    def _forward_and_log_det_jacobian(self, x, yy, nlf0=None, nlf1=None, iso=None, cam=None):
        if self._last_layer:
            x = tf.reshape(x, (-1, self.i0, self.i1, self.ic))
            yy = tf.reshape(yy, (-1, self.i0, self.i1, self.ic))

        log_scale = sdn_model_params_ex5(yy, iso, self.gain_init, cam, self.param_inits)
        log_scale = tf.tanh(log_scale)

        y = x
        if log_scale is not None:
            y *= tf.exp(log_scale)
        if log_scale is None:
            log_abs_det_J = tf.constant(0., dtype=x.dtype, name="fldj")
        else:
            log_abs_det_J = tf.reduce_sum(log_scale, axis=[1, 2, 3])
        return y, log_abs_det_J

    def _inverse_and_log_det_jacobian(self, y, yy, nlf0=None, nlf1=None, iso=None, cam=None):
        log_scale = sdn_model_params_ex5(yy, iso, self.gain_init, cam, self.param_inits)
        log_scale = tf.tanh(log_scale)
        # tf.summary.histogram('sdn/scale', scale)

        tf.summary.scalar(self.name + '_log_scale_mean', tf.reduce_mean(log_scale))
        tf.summary.scalar(self.name + '_log_scale_min', tf.reduce_min(log_scale))
        tf.summary.scalar(self.name + '_log_scale_max', tf.reduce_max(log_scale))

        x = y
        if log_scale is not None:
            x /= tf.exp(log_scale)
        if log_scale is None:
            log_abs_det_J_inv = tf.constant(0., dtype=y.dtype, name="ildj")
        else:
            log_abs_det_J_inv = - tf.reduce_sum(log_scale, axis=[1, 2, 3])
        if self._last_layer:
            return tf.layers.flatten(x), log_abs_det_J_inv
        return x, log_abs_det_J_inv
