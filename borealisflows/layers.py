# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import warnings

import numpy as np
import scipy
import tensorflow as tf

from borealisflows.matrix_param import *
from borealisflows.utils import int_shape

tfd = tf.contrib.distributions
tfb = tfd.bijectors
layers = tf.contrib.layers

# Settings
DTYPE = tf.float32
NP_DTYPE = np.float32


class LeakyReLU(tfb.Bijector):
    def __init__(
            self, alpha=1.0, learnable_alpha=True, validate_args=False,
            layer_id=0, name='leaky_relu'):
        super(LeakyReLU, self).__init__(
            validate_args=validate_args,
            is_constant_jacobian=False,
            forward_min_event_ndims=0,
            name=name)
        self.alpha = tf.convert_to_tensor(alpha, dtype=DTYPE)
        if learnable_alpha:
            with tf.variable_scope(name):
                self.alpha = tf.abs(tf.get_variable(
                    'alpha_{}'.format(layer_id),
                    initializer=self.alpha))

    def _forward(self, x):
        return tf.where(tf.greater_equal(x, 0), x, 1. / self.alpha * x)

    def _inverse(self, y):
        return tf.where(tf.greater_equal(y, 0), y, self.alpha * y)

    def _inverse_log_det_jacobian(self, y):
        I = tf.ones_like(y)
        J_inv = tf.where(tf.greater_equal(y, 0), I, self.alpha * I)
        log_abs_det_J_inv = tf.log(tf.abs(J_inv))
        # return tf.layers.flatten(log_abs_det_J_inv)
        return tf.reduce_sum(log_abs_det_J_inv, axis=[1, 2, 3])

    def _forward_log_det_jacobian(self, x):
        I = tf.ones_like(x)
        J = tf.where(tf.greater_equal(x, 0), I, 1. / self.alpha * I)
        # abs is actually redundant here, since this det Jacobian is > 0
        log_abs_det_J = tf.log(tf.abs(J))
        # return tf.layers.flatten(log_abs_det_J)
        return tf.reduce_sum(log_abs_det_J, axis=[1, 2, 3])

    def _forward_and_log_det_jacobian(self, x):
        y = self._forward(x)
        log_abs_det_J = self._forward_log_det_jacobian(x)
        return y, log_abs_det_J

    def _inverse_and_log_det_jacobian(self, y):
        x = self._inverse(y)
        log_abs_det_J_inv = self._inverse_log_det_jacobian(y)
        return x, log_abs_det_J_inv


class Conv2d1x1(tfb.Bijector):
    def __init__(
            self, x_shape, bias=True, decomp='NONE', last_layer=False,
            validate_args=False, layer_id=0, order=0, name='conv2d_1x1'):
        super(Conv2d1x1, self).__init__(
            validate_args=validate_args,
            is_constant_jacobian=True,
            forward_min_event_ndims=1,
            name=name)
        self.x_shape = x_shape
        self.i0, self.i1, self.ic = x_shape
        self._bias = bias
        self._decomp = decomp
        self.id = layer_id
        self._last_layer = last_layer
        self._order = order
        self._init_weights()

    def _init_weights(self):
        w_shape = [self.ic, self.ic]
        self.w_shape = w_shape
        np_w = scipy.linalg.qr(np.random.randn(*w_shape))[0].astype('float32')
        with tf.variable_scope(self.name):
            self.A_param = matrix_param(self._decomp,
                                        'conv2d_1x1_{}_{}'.format(
                                            self.id, self._order),
                                        np_w, DTYPE)

            if self._bias:
                self.b = tf.get_variable(
                    'b_filters_{}_{}'.format(self.id, self._order),
                    self.x_shape[-1:], dtype=DTYPE,
                    initializer=tf.zeros_initializer)

    def _forward(self, x):
        if self._last_layer:
            x = tf.reshape(x, (-1, self.i0, self.i1, self.ic))
        if self._bias:
            x = tf.subtract(x, self.b)
        _w = tf.reshape(self.A_param['A_inv'], [1, 1] + self.w_shape)
        y = tf.nn.conv2d(x, _w, [1, 1, 1, 1], 'SAME', data_format='NHWC')
        return y

    def _inverse(self, y):
        _w = tf.reshape(self.A_param['A'], [1, 1] + self.w_shape)
        x = tf.nn.conv2d(y, _w, [1, 1, 1, 1], 'SAME', data_format='NHWC')
        if self._bias:
            x = tf.add(x, self.b)
        if self._last_layer:
            x = tf.reshape(x, (-1, self.i0 * self.i1 * self.ic))
        return x

    def _forward_log_det_jacobian(self, x):
        return -self._inverse_log_det_jacobian(None)

    def _inverse_log_det_jacobian(self, y):
        return self.A_param['log_abs_det'] * (self.i0 * self.i1)

    def _forward_and_log_det_jacobian(self, x):
        y = self._forward(x)
        log_abs_det_J = self._forward_log_det_jacobian(x)
        return y, log_abs_det_J

    def _inverse_and_log_det_jacobian(self, y):
        x = self._inverse(y)
        log_abs_det_J_inv = self._inverse_log_det_jacobian(y)
        return x, log_abs_det_J_inv

    def _inverse_and_log_det_jacobian___(self, z, y):
        x = self._inverse(z)
        log_abs_det_J_inv = self._inverse_log_det_jacobian(z)
        return x, log_abs_det_J_inv


# --- BatchNorm Bijector ---
class BatchNorm(tfb.Bijector):
    def __init__(
            self, x_shape, is_training, eps=1e-4, decay=0.1,
            validate_args=False, name="batch_norm"):
        super(BatchNorm, self).__init__(
            forward_min_event_ndims=1, validate_args=validate_args, name=name)
        self.is_training = is_training
        self.eps = eps
        self.decay = decay
        # self.x_shape = x_shape
        self.i0, self.i1, self.ic = x_shape
        # self._vars_created = False
        self._create_vars()

    # def _create_vars(self, x):
    #     x_shape = x.get_shape().as_list()
    #     self.i0, self.i1 = x_shape[1], x_shape[2]
    #     with tf.variable_scope(self.name):
    #         import pdb; pdb.set_trace()
    #         self.train_m = tf.get_variable(
    #             'mean', [x_shape[-1]], dtype=DTYPE,
    #             initializer=tf.constant_initializer(0.), trainable=False)
    #         self.train_v = tf.get_variable(
    #             'var', [x_shape[-1]], dtype=DTYPE,
    #             initializer=tf.constant_initializer(1.), trainable=False)
    #     self._vars_created = True

    def _create_vars(self):
        with tf.variable_scope(self.name):
            self.train_m = tf.get_variable(
                'mean', [self.ic], dtype=DTYPE,
                initializer=tf.constant_initializer(0.), trainable=False)
            self.train_v = tf.get_variable(
                'var', [self.ic], dtype=DTYPE,
                initializer=tf.constant_initializer(1.), trainable=False)

    def _normalize(self, x, training=True):
        if training:
            # statistics of current minibatch
            m, v = tf.nn.moments(x, axes=[0, 1, 2])
            update_train_m = tf.assign_sub(
                self.train_m, self.decay * (self.train_m - m))
            update_train_v = tf.assign_sub(
                self.train_v, self.decay * (self.train_v - v))
            # normalize using current minibatch statistics
            with tf.control_dependencies([update_train_m, update_train_v]):
                x_hat = (x - m) / tf.sqrt(v + self.eps)
        else:
            x_hat = (x - self.train_m) / tf.sqrt(self.train_v + self.eps)
        return x_hat

    def _denormalize(self, x):
        return x * tf.sqrt(self.train_v + self.eps) + self.train_m

    def _forward(self, x):
        # if not self._vars_created:
        #     self._create_vars(x)
        return self._denormalize(x)

    def _inverse(self, y):
        # if not self._vars_created:
        #     self._create_vars(y)
        self.m, self.v = tf.nn.moments(y, axes=[0, 1, 2])
        return tf.cond(tf.equal(self.is_training, tf.constant(True)),
                       lambda: self._normalize(y, True),
                       lambda: self._normalize(y, False))

    def _log_det_jacobian(self, x, training=True):
        if training:
            # at training time, the log_det_jacobian is computed from
            # statistics of the current minibatch.
            _, v = tf.nn.moments(x, axes=[0, 1, 2])
            return tf.reduce_sum(
                -0.5 * tf.log(v + self.eps)) * self.i0 * self.i1
        else:
            return tf.reduce_sum(
                -0.5 * tf.log(self.train_v + self.eps)) * self.i0 * self.i1

    def _inverse_log_det_jacobian(self, y):
        # if not self._vars_created:
        #     self._create_vars(y)
        return tf.cond(tf.equal(self.is_training, tf.constant(True)),
                       lambda: self._log_det_jacobian(y, True),
                       lambda: self._log_det_jacobian(y, False))

    def _forward_log_det_jacobian(self, x):
        # if not self._vars_created:
        #     self._create_vars(x)
        return -self._log_det_jacobian(x, training=False)

    def _forward_and_log_det_jacobian(self, x):
        y = self._forward(x)
        log_abs_det_J = self._forward_log_det_jacobian(x)
        return y, log_abs_det_J

    def _inverse_and_log_det_jacobian(self, y):
        x = self._inverse(y)
        log_abs_det_J_inv = self._inverse_log_det_jacobian(y)
        return x, log_abs_det_J_inv


# --- Real-NVP ---
class AffineCoupling(tfb.Bijector):

    def __init__(
            self,
            x_shape,
            shift_and_log_scale_fn,
            layer_id=0,
            last_layer=False,
            validate_args=False,
            name="real_nvp"):
        super(AffineCoupling, self).__init__(
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

    def _forward(self, x):
        if self._last_layer:
            x = tf.reshape(x, (-1, self.i0, self.i1, self.ic))
        # Performs un-shift and un-scale.
        x0 = x[:, :, :, :self.ic // 2]
        x1 = x[:, :, :, self.ic // 2:]
        shift, log_scale = self._shift_and_log_scale_fn(x0)
        tf.summary.histogram('uncond. shift (forward)', shift)
        tf.summary.histogram('uncond. log_scale (forward)', log_scale)
        log_scale = self.scale * tf.tanh(log_scale)
        y1 = x1
        if shift is not None:
            y1 -= shift
        if log_scale is not None:
            y1 *= tf.exp(-log_scale)
        y = tf.concat([x0, y1], axis=-1)
        return y

    def _inverse(self, y):
        # Performs scale and shift.
        y0 = y[:, :, :, :self.ic // 2]
        y1 = y[:, :, :, self.ic // 2:]
        shift, log_scale = self._shift_and_log_scale_fn(y0)
        # tf.summary.histogram('uncond/shift', shift)
        # tf.summary.histogram('uncond/logscale', log_scale)
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

    def _forward_log_det_jacobian(self, x):
        if self._last_layer:
            x = tf.reshape(x, (-1, self.i0, self.i1, self.ic))
        x0 = x[:, :, :, :self.ic // 2]
        _, log_scale = self._shift_and_log_scale_fn(x0)
        log_scale = self.scale * tf.tanh(log_scale)
        # tf.summary.histogram('uncond/logscale', log_scale)
        if log_scale is None:
            return tf.constant(0., dtype=x.dtype, name="fldj")
        # return -tf.reduce_sum(tf.log(log_scale), axis=[1, 2, 3])
        return -tf.reduce_sum(log_scale, axis=[1, 2, 3])

    def _inverse_log_det_jacobian(self, z):
        z0 = z[:, :, :, :self.ic // 2]
        _, log_scale = self._shift_and_log_scale_fn(z0)
        log_scale = self.scale * tf.tanh(log_scale)
        # tf.summary.histogram('uncond/logscale', log_scale)
        if log_scale is None:
            return tf.constant(0., dtype=z.dtype, name="ildj")
        # return tf.reduce_sum(tf.log(log_scale), axis=[1, 2, 3])
        return tf.reduce_sum(log_scale, axis=[1, 2, 3])

    def _forward_and_log_det_jacobian(self, x):
        if self._last_layer:
            x = tf.reshape(x, (-1, self.i0, self.i1, self.ic))
        # Performs un-shift and un-scale.
        x0 = x[:, :, :, :self.ic // 2]
        x1 = x[:, :, :, self.ic // 2:]
        shift, log_scale = self._shift_and_log_scale_fn(x0)
        # tf.summary.histogram('uncond/shift', shift)
        # tf.summary.histogram('uncond/logscale', log_scale)
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

    def _inverse_and_log_det_jacobian(self, y):
        # Performs scale and shift.
        y0 = y[:, :, :, :self.ic // 2]
        y1 = y[:, :, :, self.ic // 2:]
        shift, log_scale = self._shift_and_log_scale_fn(y0)
        # tf.summary.histogram('uncond/shift', shift)
        # tf.summary.histogram('uncond/logscale', log_scale)
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


def batch_norm(x, training=True, eps=1e-4, decay=0.1, name="batch_norm"):
    x_shape = x.get_shape().as_list()
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        assert len(x_shape) == 2 or len(x_shape) == 4
        train_m = tf.get_variable(
            'mean', [x_shape[-1]], dtype=DTYPE,
            initializer=tf.zeros_initializer, trainable=False)
        train_v = tf.get_variable(
            'var', [x_shape[-1]], dtype=DTYPE,
            initializer=tf.ones_initializer, trainable=False)
        if training:
            # statistics of current minibatch
            if len(x_shape) == 2:
                m, v = tf.nn.moments(x, axes=[0])
            elif len(x_shape) == 4:
                m, v = tf.nn.moments(x, axes=[0, 1, 2])
            update_train_m = tf.assign_sub(train_m, decay * (train_m - m))
            update_train_v = tf.assign_sub(train_v, decay * (train_v - v))
            # normalize using current minibatch statistics
            with tf.control_dependencies([update_train_m, update_train_v]):
                x_hat = (x - m) / tf.sqrt(v + eps)
        else:
            x_hat = (x - train_m) / tf.sqrt(train_v + eps)
    return x_hat


def real_nvp_default_template(
        x_shape,
        is_training,
        hidden_layers,
        shift_only=False,
        activation=tf.nn.relu,
        name=None,
        *args,
        **kwargs):
    template_name = "real_nvp_default_template"
    with tf.name_scope(name, template_name):
        def _fn(x):
            """Fully connected MLP parameterized via `real_nvp_template`."""
            i0, i1, ic = x_shape
            ic = int(ic / 2)
            output_units = i0 * i1 * ic
            num_output = (1 if shift_only else 2) * output_units
            x_flat = tf.reshape(x, (-1, i0 * i1 * ic))
            for i, units in enumerate(hidden_layers):
                x_flat = tf.layers.dense(inputs=x_flat, units=units)
                x_flat = tf.cond(
                    tf.equal(is_training, tf.constant(True)),
                    lambda: batch_norm(
                        x_flat, True, name='bn_nn_{}'.format(i)),
                    lambda: batch_norm(
                        x_flat, False, name='bn_nn_{}'.format(i)))
                x_flat = activation(x_flat)
            # the last layer is initialized as all zeros, effectively making
            # the affine coupling layer an identity mapping at the beginning
            x_flat = tf.layers.dense(
                inputs=x_flat,
                units=num_output,
                kernel_initializer=tf.zeros_initializer(),
                bias_initializer=tf.zeros_initializer(),
                activation=None,
                *args,
                **kwargs)
            x = tf.reshape(
                x_flat, (-1, i0, i1, int(num_output / output_units * ic)))
            if shift_only:
                return x, None
            shift, log_scale = tf.split(x, 2, axis=-1)
            # shift = x[:, :, :, 0::2]
            # log_scale = x[:, :, :, 1::2]
            return shift, log_scale
        return tf.make_template(template_name, _fn)


def real_nvp_conv_template(
        x_shape,
        is_training,
        width=512,
        shift_only=False,
        activation=tf.nn.relu,
        name=None,
        *args,
        **kwargs):
    template_name = "real_nvp_conv_template"
    with tf.name_scope(name, template_name):
        def _fn(x):
            """Fully connected MLP parameterized via `real_nvp_template`."""
            i0, i1, ic = x_shape
            ic = int(ic / 2)
            num_output = (1 if shift_only else 2) * ic

            x = conv2d('l_1', x, width)
            with warnings.catch_warnings():  # ignore np.asscalar() deprecated raised by tf.constant(True)
                warnings.simplefilter("ignore")
                x = tf.cond(
                    tf.equal(is_training, tf.constant(True)),
                    lambda: batch_norm(
                        x, True, name='bn_nvp_conv_1'),
                    lambda: batch_norm(
                        x, False, name='bn_nvp_conv_1'))
            x = activation(x)

            x = conv2d('l_2', x, width, filter_size=[1, 1])
            with warnings.catch_warnings():  # ignore np.asscalar() deprecated raised by tf.constant(True)
                warnings.simplefilter("ignore")
                x = tf.cond(
                    tf.equal(is_training, tf.constant(True)),
                    lambda: batch_norm(
                        x, True, name='bn_nvp_conv_2'),
                    lambda: batch_norm(
                        x, False, name='bn_nvp_conv_2'))
            x = activation(x)

            x = conv2d_zeros('l_last', x, num_output)
            if shift_only:
                return x, None
            shift, log_scale = tf.split(x, 2, axis=-1)
            # shift = x[:, :, :, 0::2]
            # log_scale = x[:, :, :, 1::2]
            return shift, log_scale
        return tf.make_template(template_name, _fn)


def real_nvp_conv_template_iso(
        x_shape,
        is_training,
        width=128,
        shift_only=False,
        activation=tf.nn.relu,
        name=None,
        *args,
        **kwargs):
    template_name = "real_nvp_conv_template_iso"
    with tf.name_scope(name, template_name):
        def _fn(x, iso):
            """Fully connected MLP parameterized via `real_nvp_template`."""
            i0, i1, ic = x_shape
            ic = int(ic / 2)
            num_output = (1 if shift_only else 2) * ic

            x = conv2d_iso('l_1', x, width, iso)
            with warnings.catch_warnings():  # ignore np.asscalar() deprecated raised by tf.constant(True)
                warnings.simplefilter("ignore")
                x = tf.cond(
                    tf.equal(is_training, tf.constant(True)),
                    lambda: batch_norm(
                        x, True, name='bn_nvp_conv_1'),
                    lambda: batch_norm(
                        x, False, name='bn_nvp_conv_1'))
            x = activation(x)

            x = conv2d_iso('l_2', x, width, iso, filter_size=[1, 1])
            with warnings.catch_warnings():  # ignore np.asscalar() deprecated raised by tf.constant(True)
                warnings.simplefilter("ignore")
                x = tf.cond(
                    tf.equal(is_training, tf.constant(True)),
                    lambda: batch_norm(
                        x, True, name='bn_nvp_conv_2'),
                    lambda: batch_norm(
                        x, False, name='bn_nvp_conv_2'))
            x = activation(x)

            x = conv2d_zeros('l_last', x, num_output)
            if shift_only:
                return x, None
            shift, log_scale = tf.split(x, 2, axis=-1)
            # shift = x[:, :, :, 0::2]
            # log_scale = x[:, :, :, 1::2]
            return shift, log_scale
        return tf.make_template(template_name, _fn)


# conv wrappers from Glow
def default_initializer(std=0.05):
    return tf.random_normal_initializer(0., std)


def add_edge_padding(x, filter_size):
    """Slow way to add edge padding"""
    assert filter_size[0] % 2 == 1
    if filter_size[0] == 1 and filter_size[1] == 1:
        return x
    a = (filter_size[0] - 1) // 2  # vertical padding size
    b = (filter_size[1] - 1) // 2  # horizontal padding size
    if True:
        x = tf.pad(x, [[0, 0], [a, a], [b, b], [0, 0]])
        name = "_".join([str(dim) for dim in [a, b, *int_shape(x)[1:3]]])
        pads = tf.get_collection(name)
        if not pads:
            pad = np.zeros([1] + int_shape(x)[1:3] + [1], dtype='float32')
            pad[:, :a, :, 0] = 1.
            pad[:, -a:, :, 0] = 1.
            pad[:, :, :b, 0] = 1.
            pad[:, :, -b:, 0] = 1.
            pad = tf.convert_to_tensor(pad)
            tf.add_to_collection(name, pad)
        else:
            pad = pads[0]
        pad = tf.tile(pad, [tf.shape(x)[0], 1, 1, 1])
        x = tf.concat([x, pad], axis=3)
    else:
        pad = tf.pad(tf.zeros_like(x[:, :, :, :1]) - 1,
                     [[0, 0], [a, a], [b, b], [0, 0]]) + 1
        x = tf.pad(x, [[0, 0], [a, a], [b, b], [0, 0]])
        x = tf.concat([x, pad], axis=3)
    return x


def conv2d(
        name, x, width, filter_size=[3, 3], stride=[1, 1], pad="SAME",
        do_weightnorm=False, context1d=None, skip=1, edge_bias=False):  # Abdo: changed edge_bias to False
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):  # Abdo: added AUTO_REUSE
        if edge_bias and pad == "SAME":
            x = add_edge_padding(x, filter_size)
            pad = 'VALID'

        n_in = int(x.get_shape()[3])

        stride_shape = [1] + stride + [1]
        filter_shape = filter_size + [n_in, width]
        w = tf.get_variable("W", filter_shape, tf.float32,
                            initializer=default_initializer(width / 512 * 0.05))  # abdo

        if do_weightnorm:
            w = tf.nn.l2_normalize(w, [0, 1, 2])
        if skip == 1:
            x = tf.nn.conv2d(x, w, stride_shape, pad, data_format='NHWC')
        else:
            assert stride[0] == 1 and stride[1] == 1
            x = tf.nn.atrous_conv2d(x, w, skip, pad)
        x += tf.get_variable("b", [1, 1, 1, width],
                             initializer=tf.zeros_initializer())
        # if context1d != None:
        #     x += tf.reshape(linear("context", context1d,
        #                            width), [-1, 1, 1, width])
    return x


def conv2d_iso(
        name, x, width, iso, filter_size=[3, 3], stride=[1, 1], pad="SAME",
        do_weightnorm=False, skip=1, edge_bias=False):  # Abdo: changed edge_bias to False
    with tf.variable_scope(name):
        if edge_bias and pad == "SAME":
            x = add_edge_padding(x, filter_size)
            pad = 'VALID'

        n_in = int(x.get_shape()[3])

        # init_sd = 0.05 * (512.0 / width)
        init_sd = 0.05

        stride_shape = [1] + stride + [1]
        filter_shape = filter_size + [n_in, width]
        b1 = tf.get_variable("B1", filter_shape, tf.float32, initializer=default_initializer(init_sd))
        b2 = tf.get_variable("B2", filter_shape, tf.float32, initializer=default_initializer(init_sd))
        w = b1 * iso[0] + b2
        # w = tf.nn.conv2d(iso, b1) + b2
        # w = tf.get_variable("W", filter_shape, tf.float32, initializer=default_initializer())

        if do_weightnorm:
            w = tf.nn.l2_normalize(w, [0, 1, 2])
        if skip == 1:
            x = tf.nn.conv2d(x, w, stride_shape, pad, data_format='NHWC')
        else:
            assert stride[0] == 1 and stride[1] == 1
            x = tf.nn.atrous_conv2d(x, w, skip, pad)
        # x += tf.get_variable("b", [1, 1, 1, width], initializer=tf.zeros_initializer())
        c1 = tf.get_variable("C1", [1, 1, 1, width], tf.float32, initializer=default_initializer(init_sd))
        c2 = tf.get_variable("C2", [1, 1, 1, width], tf.float32, initializer=default_initializer(init_sd))
        x += c1 * iso[0] + c2
    return x


def conv2d_zeros(
        name, x, width, filter_size=[3, 3], stride=[1, 1], pad="SAME",
        logscale_factor=3, skip=1, edge_bias=True):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):  # Abdo: added AUTO_REUSE
        if edge_bias and pad == "SAME":
            x = add_edge_padding(x, filter_size)
            pad = 'VALID'

        n_in = int(x.get_shape()[3])
        stride_shape = [1] + stride + [1]
        filter_shape = filter_size + [n_in, width]
        w = tf.get_variable("W", filter_shape, tf.float32,
                            initializer=tf.zeros_initializer())
        if skip == 1:
            x = tf.nn.conv2d(x, w, stride_shape, pad, data_format='NHWC')
        else:
            assert stride[0] == 1 and stride[1] == 1
            x = tf.nn.atrous_conv2d(x, w, skip, pad)
        x += tf.get_variable("b", [1, 1, 1, width],
                             initializer=tf.zeros_initializer())
        x *= tf.exp(
            tf.get_variable(
                "logs", [1, width], initializer=tf.zeros_initializer()) * logscale_factor)
    return x


def linear_zeros(name, x, width, logscale_factor=3):
    """Linear layer with zero init"""
    with tf.variable_scope(name):
        n_in = int(x.get_shape()[1])
        w = tf.get_variable("W", [n_in, width], tf.float32,
                            initializer=tf.zeros_initializer())
        x = tf.matmul(x, w)
        x += tf.get_variable("b", [1, width],
                             initializer=tf.zeros_initializer())
        x *= tf.exp(tf.get_variable(
            "logs", [1, width],
            initializer=tf.zeros_initializer()) * logscale_factor)
        return x
