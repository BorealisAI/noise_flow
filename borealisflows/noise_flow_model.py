# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from borealisflows.noise_flow_layers.AffineCouplingCondXY import AffineCouplingCondXY
from borealisflows.noise_flow_layers.AffineCouplingCondXYG import AffineCouplingCondXYG
from borealisflows.noise_flow_layers.AffineCouplingCondY import AffineCouplingCondY
from borealisflows.noise_flow_layers.AffineCouplingCondYG import AffineCouplingCondYG
from borealisflows.noise_flow_layers.AffineCouplingGain import AffineCouplingGain
from borealisflows.noise_flow_layers.AffineCouplingGainEx1 import AffineCouplingGainEx1
from borealisflows.noise_flow_layers.AffineCouplingGainEx2 import AffineCouplingGainEx2
from borealisflows.noise_flow_layers.AffineCouplingGainEx3 import AffineCouplingGainEx3
from borealisflows.noise_flow_layers.AffineCouplingGainEx4 import AffineCouplingGainEx4
from borealisflows.noise_flow_layers.AffineCouplingSdn import AffineCouplingSdn
from borealisflows.noise_flow_layers.AffineCouplingSdnEx1 import AffineCouplingSdnEx1
from borealisflows.noise_flow_layers.AffineCouplingSdnEx2 import AffineCouplingSdnEx2
from borealisflows.noise_flow_layers.AffineCouplingSdnEx3 import AffineCouplingSdnEx3
from borealisflows.noise_flow_layers.AffineCouplingSdnEx4 import AffineCouplingSdnEx4
from borealisflows.noise_flow_layers.AffineCouplingSdnEx5 import AffineCouplingSdnEx5
from borealisflows.noise_flow_layers.AffineCouplingSdnEx6 import AffineCouplingSdnEx6
from borealisflows.noise_flow_layers.AffineCouplingSdnGain import AffineCouplingSdnGain
from borealisflows.noise_flow_layers.AffineCouplingCamSdn import AffineCouplingCamSdn
from borealisflows.noise_flow_layers.AffineCouplingFitSdnGain2 import AffineCouplingFitSdnGain2
from borealisflows.layers import real_nvp_conv_template_iso
from borealisflows.layers import AffineCoupling
from borealisflows.layers import real_nvp_conv_template
from borealisflows.layers import Conv2d1x1
from borealisflows.layers import conv2d_zeros
from borealisflows.utils import int_shape
from borealisflows.utils import squeeze2d
from borealisflows.utils import unsqueeze2d

tfd = tfp.distributions
tfb = tfp.bijectors


class NoiseFlow(object):

    def __init__(self, x_shape, is_training, hps=None):
        self.x_shape = x_shape
        self.depth = hps.depth
        self.n_levels = hps.n_levels
        self._is_training = is_training
        self.hps = hps
        self.model = self.define_flow_structure()

    def define_flow_structure(self):
        model = []
        x_shape = self.x_shape
        for i in range(self.n_levels):
            if self.hps.squeeze_factor != 1:
                if i == 0:
                    x_shape = [x_shape[0] // 2, x_shape[1] // 2, x_shape[2] * 4]
                else:
                    x_shape = [x_shape[0] // 2, x_shape[1] // 2, x_shape[2] * 2]
            if self.hps.arch is not None:
                model.append(self.noise_flow_arch('level{}'.format(i), x_shape,
                                                  self.hps.flow_permutation, self.hps.arch))
            else:
                model.append(self.revnet2d('level{}'.format(i), x_shape,
                                           self.hps.flow_permutation))
        return model

    def noise_flow_arch(self, name, x_shape, flow_permutation, arch):
        arch_lyrs = arch.split('|')  # e.g., unc|sdn|unc|gain|unc
        bijectors = []
        with tf.variable_scope(name):
            for i, lyr in enumerate(arch_lyrs):
                with tf.variable_scope('bijector{}'.format(i)):
                    is_last_layer = False

                    if lyr == 'unc':
                        if flow_permutation == 0:
                            print('|-tfb.Permute')
                            bijectors.append(
                                tfb.Permute(
                                    permutation=list(range(x_shape[-1]))[::-1]))
                        elif flow_permutation == 1:
                            print('|-Conv2d1x1')
                            bijectors.append(
                                Conv2d1x1(x_shape, layer_id=i, bias=False,
                                          decomp=self.hps.decomp,
                                          name='Conv2d_1x1_{}'.format(i)))
                        else:
                            print('|-No permutation specified. Not using any.')
                            # raise Exception("Flow permutation not understood")

                        print('|-AffineCoupling')
                        bijectors.append(
                            AffineCoupling(
                                name='unc_%d' % i,
                                last_layer=is_last_layer,
                                x_shape=x_shape,
                                shift_and_log_scale_fn=real_nvp_conv_template(
                                    x_shape=x_shape,
                                    is_training=self._is_training,
                                    width=self.hps.width)))

                    elif lyr == 'sdn':
                        print('|-AffineCouplingSdn')
                        bijectors.append(
                            AffineCouplingSdn(
                                name='sdn_%d' % i,
                                last_layer=False,
                                x_shape=x_shape,
                                shift_and_log_scale_fn=None
                            )
                        )
                    elif lyr == 'sdn1':
                        print('|-AffineCouplingSdnEx1')
                        bijectors.append(
                            AffineCouplingSdnEx1(
                                name='sdn_%d' % i,
                                last_layer=False,
                                x_shape=x_shape,
                                shift_and_log_scale_fn=None
                            )
                        )
                    elif lyr == 'sdn2':
                        print('|-AffineCouplingSdnEx2')
                        bijectors.append(
                            AffineCouplingSdnEx2(
                                name='sdn_%d' % i,
                                last_layer=False,
                                x_shape=x_shape,
                                shift_and_log_scale_fn=None,
                                gain_init=self.hps.gain_init
                            )
                        )
                    elif lyr == 'sdn3':
                        print('|-AffineCouplingSdnEx3')
                        bijectors.append(
                            AffineCouplingSdnEx3(
                                name='sdn_%d' % i,
                                last_layer=False,
                                x_shape=x_shape,
                                shift_and_log_scale_fn=None,
                                gain_init=self.hps.gain_init
                            )
                        )
                    elif lyr == 'sdn4':
                        print('|-AffineCouplingSdnEx4')
                        bijectors.append(
                            AffineCouplingSdnEx4(
                                name='sdn_%d' % i,
                                last_layer=False,
                                x_shape=x_shape,
                                shift_and_log_scale_fn=None,
                                gain_init=self.hps.gain_init
                            )
                        )
                    elif lyr == 'sdn5':
                        print('|-AffineCouplingSdnEx5')
                        bijectors.append(
                            AffineCouplingSdnEx5(
                                name='sdn_%d' % i,
                                last_layer=False,
                                x_shape=x_shape,
                                shift_and_log_scale_fn=None,
                                gain_init=self.hps.gain_init,
                                param_inits=self.hps.param_inits
                            )
                        )
                    elif lyr == 'sdn6':
                        print('|-AffineCouplingSdnEx6')
                        bijectors.append(
                            AffineCouplingSdnEx6(
                                name='sdn_%d' % i,
                                last_layer=False,
                                x_shape=x_shape,
                                shift_and_log_scale_fn=None,
                                gain_init=self.hps.gain_init,
                                param_inits=self.hps.param_inits
                            )
                        )
                    elif lyr == 'gain':
                        print('|-AffineCouplingGain')
                        bijectors.append(
                            AffineCouplingGain(
                                name='gain_%d' % i,
                                last_layer=False,
                                x_shape=x_shape,
                                shift_and_log_scale_fn=None
                            )
                        )
                    elif lyr == 'gain1':
                        print('|-AffineCouplingGainEx1')
                        bijectors.append(
                            AffineCouplingGainEx1(
                                name='gain_%d' % i,
                                last_layer=False,
                                x_shape=x_shape,
                                shift_and_log_scale_fn=None
                            )
                        )
                    elif lyr == 'gain2':
                        print('|-AffineCouplingGainEx2')
                        bijectors.append(
                            AffineCouplingGainEx2(
                                name='gain_%d' % i,
                                last_layer=False,
                                x_shape=x_shape,
                                shift_and_log_scale_fn=None,
                                gain_init=self.hps.gain_init
                            )
                        )
                    elif lyr == 'gain3':
                        print('|-AffineCouplingGainEx3')
                        bijectors.append(
                            AffineCouplingGainEx3(
                                name='gain_%d' % i,
                                last_layer=False,
                                x_shape=x_shape,
                                shift_and_log_scale_fn=None
                            )
                        )
                    elif lyr == 'gain4':
                        print('|-AffineCouplingGainEx4')
                        bijectors.append(
                            AffineCouplingGainEx4(
                                name='gain_%d' % i,
                                last_layer=False,
                                x_shape=x_shape,
                                shift_and_log_scale_fn=None,
                                gain_init=self.hps.gain_init
                            )
                        )
        return bijectors

    def revnet2d(self, name, x_shape, flow_permutation):
        """Affine coupling"""
        print('sidd_cond = %s' % self.hps.sidd_cond)
        bijectors = []
        with tf.variable_scope(name):
            # append an SDN layer (2nd degree) next to base measure
            if self.hps.append_sdn2:
                with tf.variable_scope('bijector_sdn2'):
                    print('|-AffineCouplingFitSdnGain2')
                    bijectors.append(
                        AffineCouplingFitSdnGain2(
                            name='ac_fitSdnGain2_%d' % self.depth,
                            last_layer=False,
                            x_shape=x_shape,
                            shift_and_log_scale_fn=None
                        )
                    )
            # append an SDN layer next to base measure
            if self.hps.append_sdn_first:
                with tf.variable_scope('bijector_sdn'):
                    print('|-AffineCouplingFitSdnGain')
                    bijectors.append(
                        AffineCouplingSdnGain(
                            name='ac_fitSdnGain_%d' % self.depth,
                            last_layer=False,
                            x_shape=x_shape,
                            shift_and_log_scale_fn=None
                        )
                    )
            # append a y-conditional layer
            if self.hps.append_cY:
                with tf.variable_scope('bijector_cy'):
                    print('|-AffineCouplingCondY')
                    bijectors.append(
                        AffineCouplingCondY(
                            name='ac_cY_first',
                            last_layer=False,
                            x_shape=x_shape,
                            shift_and_log_scale_fn=real_nvp_conv_template(
                                x_shape=x_shape[:-1] + [x_shape[-1] * 2],  # double outputs
                                is_training=self._is_training,
                                width=self.hps.width)
                        )
                    )
            for i in range(self.depth):
                with tf.variable_scope('bijector{}'.format(i)):
                    # is_last_layer = True if i == self.depth - 1 else False
                    is_last_layer = False

                    # bijectors.append(
                    #     BatchNorm(x_shape, is_training=self._is_training,
                    #               name='batch_norm{}'.format(i)))

                    if flow_permutation == 0:
                        print('|-tfb.Permute')
                        bijectors.append(
                            tfb.Permute(
                                permutation=list(range(x_shape[-1]))[::-1]))
                    elif flow_permutation == 1:
                        print('|-Conv2d1x1')
                        bijectors.append(
                            Conv2d1x1(x_shape, layer_id=i, bias=False,
                                      decomp=self.hps.decomp,
                                      name='Conv2d_1x1_{}'.format(i)))
                    else:
                        print('|-No permutation specified. Not using any.')
                        # raise Exception("Flow permutation not understood")

                    if self.hps.sidd_cond == 'condY':
                        print('|-AffineCouplingCondY')
                        bijectors.append(
                            AffineCouplingCondY(
                                name='ac_cY_%d' % i,
                                last_layer=is_last_layer,
                                x_shape=x_shape,
                                shift_and_log_scale_fn=real_nvp_conv_template(
                                    x_shape=x_shape[:-1] + [x_shape[-1] * 2],  # double outputs
                                    is_training=self._is_training,
                                    width=self.hps.width)))
                    elif self.hps.sidd_cond == 'condYG':
                        print('|-AffineCouplingCondYG')
                        bijectors.append(
                            AffineCouplingCondYG(
                                name='ac_cYG_%d' % i,
                                last_layer=is_last_layer,
                                x_shape=x_shape,
                                shift_and_log_scale_fn=real_nvp_conv_template_iso(
                                    x_shape=x_shape[:-1] + [x_shape[-1] * 2],  # double outputs
                                    is_training=self._is_training,
                                    width=self.hps.width)))
                    elif self.hps.sidd_cond == 'condXY':
                        print('|-AffineCouplingCondXY')
                        bijectors.append(
                            AffineCouplingCondXY(
                                name='ac_cXY_%d' % i,
                                last_layer=is_last_layer,
                                x_shape=x_shape,
                                shift_and_log_scale_fn=real_nvp_conv_template(
                                    x_shape=x_shape,
                                    is_training=self._is_training,
                                    width=self.hps.width)))
                    elif self.hps.sidd_cond == 'condXYG':
                        print('|-AffineCouplingCondXYG')
                        bijectors.append(
                            AffineCouplingCondXYG(
                                name='ac_cXYG_%d' % i,
                                last_layer=is_last_layer,
                                x_shape=x_shape,
                                shift_and_log_scale_fn=real_nvp_conv_template_iso(
                                    x_shape=x_shape,
                                    is_training=self._is_training,
                                    width=self.hps.width)))
                    elif self.hps.sidd_cond == 'condSDN':
                        print('|-AffineCouplingSDN')
                        bijectors.append(
                            AffineCouplingCamSdn(
                                name='ac_cSDN_%d' % i,
                                last_layer=is_last_layer,
                                x_shape=x_shape,
                                shift_and_log_scale_fn=real_nvp_conv_template(
                                    x_shape=x_shape[:-1] + [x_shape[-1] * 2],  # double outputs
                                    is_training=self._is_training,
                                    width=self.hps.width)))
                    elif self.hps.sidd_cond == 'fitSDN':
                        print('|-AffineCouplingFitSDN')
                        bijectors.append(
                            AffineCouplingSdnGain(
                                name='ac_fitSDN_%d' % i,
                                last_layer=is_last_layer,
                                x_shape=x_shape,
                                shift_and_log_scale_fn=None))
                    else:  # uncond | unc_sdn
                        print('|-AffineCoupling')
                        bijectors.append(
                            AffineCoupling(
                                name='ac_unc_%d' % i,
                                last_layer=is_last_layer,
                                x_shape=x_shape,
                                shift_and_log_scale_fn=real_nvp_conv_template(
                                    x_shape=x_shape,
                                    is_training=self._is_training,
                                    width=self.hps.width)))
            # append an SDN layer next to data
            if self.hps.append_sdn:
                with tf.variable_scope('bijector{}'.format(self.depth)):
                    print('|-AffineCouplingFitSDN')
                    bijectors.append(
                        AffineCouplingSdnGain(
                            name='ac_fitSDN_%d' % self.depth,
                            last_layer=False,
                            x_shape=x_shape,
                            shift_and_log_scale_fn=None
                        )
                    )

        return bijectors

    def inverse(self, x, objective, yy=None, nlf0=None, nlf1=None, iso=None, cam=None):
        z = x
        squeeze_factor = self.hps.squeeze_factor

        for i in range(self.n_levels):
            z = squeeze2d(z, squeeze_factor, self.hps.squeeze_type)
            if yy is not None:
                yy = squeeze2d(yy, squeeze_factor, self.hps.squeeze_type)
            for bijector in self.model[i]:
                if type(bijector) in [AffineCouplingCondY, AffineCouplingCondXY, AffineCouplingFitSdnGain2,
                                      AffineCouplingCondYG, AffineCouplingCamSdn, AffineCouplingCondXYG,
                                      AffineCouplingSdnGain, AffineCouplingSdn, AffineCouplingGain,
                                      AffineCouplingGainEx1, AffineCouplingGainEx2, AffineCouplingGainEx3,
                                      AffineCouplingSdnEx1, AffineCouplingSdnEx2, AffineCouplingSdnEx3,
                                      AffineCouplingSdnEx4, AffineCouplingGainEx4, AffineCouplingSdnEx5,
                                      AffineCouplingSdnEx6]:
                    try:
                        z, log_abs_det_J_inv = \
                            bijector._inverse_and_log_det_jacobian(z, yy, nlf0, nlf1, iso, cam)
                    except Exception as e:
                        print(e)
                        z = bijector._inverse(z, yy, nlf0, nlf1, iso, cam)
                        log_abs_det_J_inv = bijector._inverse_log_det_jacobian(z, yy, nlf0, nlf1, iso, cam)
                else:
                    try:
                        z, log_abs_det_J_inv = \
                            bijector._inverse_and_log_det_jacobian(z)
                    except Exception as e:
                        print(e)
                        z = bijector._inverse(z)
                        log_abs_det_J_inv = bijector._inverse_log_det_jacobian(z)
                tf.summary.scalar(bijector.name, tf.reduce_mean(log_abs_det_J_inv))
                objective += log_abs_det_J_inv
            if i < self.n_levels - 1:
                z, objective = split2d("pool{}".format(i), z, objective)
        return z, objective

    def forward(self, z, eps_std, yy=None, nlf0=None, nlf1=None, iso=None, cam=None):
        x = z
        for i in reversed(range(self.n_levels)):
            if i < self.n_levels - 1:
                x = split2d_reverse("pool{}".format(i), x, eps_std)
            for bijector in reversed(self.model[i]):
                if type(bijector) in [AffineCouplingCondY, AffineCouplingCondXY, AffineCouplingFitSdnGain2,
                                      AffineCouplingCondYG, AffineCouplingCamSdn, AffineCouplingCondXYG,
                                      AffineCouplingSdnGain, AffineCouplingSdn, AffineCouplingGain,
                                      AffineCouplingGainEx1, AffineCouplingGainEx2, AffineCouplingGainEx3,
                                      AffineCouplingSdnEx1, AffineCouplingSdnEx2, AffineCouplingSdnEx3,
                                      AffineCouplingSdnEx4, AffineCouplingGainEx4, AffineCouplingSdnEx5,
                                      AffineCouplingSdnEx6]:
                    x = bijector._forward(x, yy, nlf0, nlf1, iso, cam)
                else:
                    x = bijector._forward(x)
            x = unsqueeze2d(x, self.hps.squeeze_factor, self.hps.squeeze_type)
        return x

    def sample(self, y, eps_std=None, yy=None, nlf0=None, nlf1=None, iso=None, cam=None):
        """Sampling function"""
        # y_onehot = tf.cast(tf.one_hot(tf.cast(y, 'int32'), self.hps.n_y, 1, 0), 'float32')  # not used for sidd problem
        with tf.variable_scope('model', reuse=True):
            _, sample = self.prior("prior", y)
            z = sample(eps_std)
            x = self.forward(z, eps_std, yy, nlf0, nlf1, iso, cam)
            batch_average = tf.reduce_mean(x, axis=0)
            tf.summary.histogram('sample_noise', batch_average)
            tf.summary.scalar('sample_noise_std', tf.math.reduce_std(batch_average))
        return x

    def _loss(self, x, y, nlf0=None, nlf1=None, iso=None, cam=None, reuse=False):
        with tf.variable_scope('model', reuse=reuse):
            objective = tf.zeros_like(x, dtype='float32')[:, 0, 0, 0]
            # y_onehot = tf.cast(tf.one_hot(tf.cast(y, 'int32'), 1, 1, 0), 'float32')  # not used for sidd problem
            # inverse flow
            # z, objective = self.inverse(z, objective)
            if self.hps.sidd_cond is not None and self.hps.sidd_cond != 'uncond':
                z, objective = self.inverse(x, objective, yy=y, nlf0=nlf0, nlf1=nlf1, iso=iso, cam=cam)
            else:
                z, objective = self.inverse(x, objective)

            self.hps.top_shape = int_shape(z)[1:]

            # base measure
            logp, _ = self.prior("prior", x)

            log_z = logp(z)
            objective += log_z
            tf.summary.scalar("log_z", tf.reduce_mean(log_z))
            nobj = - objective
            # std. dev. of z
            mu_z, var_z = tf.nn.moments(z, [1, 2, 3])
            sd_z = tf.reduce_mean(tf.sqrt(var_z))

        return nobj, sd_z

    def loss(self, x, y, nlf0=None, nlf1=None, iso=None, cam=None, reuse=False):
        batch_average = tf.reduce_mean(x, axis=0)
        tf.summary.histogram('real_noise', batch_average)
        tf.summary.scalar('real_noise_std', tf.math.reduce_std(batch_average))

        nll, sd_z = self._loss(x=x, y=y, nlf0=nlf0, nlf1=nlf1, iso=iso, cam=cam, reuse=reuse)  # returns NLL

        tf.summary.scalar("NLL", tf.reduce_mean(nll))
        return tf.reduce_mean(nll), sd_z

    def prior(self, name, x):
        with tf.variable_scope(name):
            n_z = self.hps.x_shape[-1]
            h = tf.zeros([tf.shape(x)[0]] + self.hps.x_shape[1:3] + [2 * n_z])
            # h = tf.zeros([tf.shape(x)[0]] +
            #              self.hps.top_shape[:2] +
            #              [2 * n_z])
            pz = gaussian_diag(h[:, :, :, :n_z], h[:, :, :, n_z:])

        def logp(z1):
            objective = pz.logp(z1)
            return objective

        def sample(eps_std=None):
            if eps_std is not None:
                z = pz.sample2(pz.eps * tf.reshape(eps_std, [-1, 1, 1, 1]))
            else:
                z = pz.sample
            return z

        return logp, sample

    def get_layer_names(self):
        layer_names = []
        for i in range(self.n_levels):
            for b in self.model[i]:
                layer_names.append(b.name)
        return layer_names


def flatten_sum(logps):
    if len(logps.get_shape()) == 2:
        return tf.reduce_sum(logps, [1])
    elif len(logps.get_shape()) == 4:
        return tf.reduce_sum(logps, [1, 2, 3])
    else:
        raise Exception()


def gaussian_diag(mean, logsd):
    class o(object):
        pass

    o.mean = mean
    o.logsd = logsd
    o.eps = tf.random_normal(tf.shape(mean))
    o.sample = mean + tf.exp(logsd) * o.eps
    o.sample2 = lambda eps: mean + tf.exp(logsd) * eps
    # o.logps = lambda x: -0.5 * \
    #     (np.log(2 * np.pi) + 2. * logsd + (x - mean) ** 2 / tf.exp(2. * logsd))
    # TODO
    o.logps = lambda x: -0.5 * \
                        (np.log(2 * np.pi) + 2. * o.logsd + (x - o.mean) ** 2 / tf.exp(2. * o.logsd))
    o.logp = lambda x: flatten_sum(o.logps(x))
    o.get_eps = lambda x: (x - mean) / tf.exp(logsd)
    return o


def split2d_prior(z):
    n_z2 = int(z.get_shape()[3])
    n_z1 = n_z2
    h = conv2d_zeros("conv", z, 2 * n_z1)

    mean = h[:, :, :, 0::2]
    logs = h[:, :, :, 1::2]
    return gaussian_diag(mean, logs)


def split2d(name, z, objective=0.):
    with tf.variable_scope(name):
        n_z = int_shape(z)[3]
        z1 = z[:, :, :, :n_z // 2]
        z2 = z[:, :, :, n_z // 2:]
        pz = split2d_prior(z1)
        objective += pz.logp(z2)
        # z1 = squeeze2d(z1)
        return z1, objective


def split2d_reverse(name, z, eps_std=None):
    with tf.variable_scope(name):
        z1 = z
        # z1 = unsqueeze2d(z)
        pz = split2d_prior(z1)
        z2 = pz.sample
        if eps_std is not None:
            z2 = pz.sample2(pz.eps * tf.reshape(eps_std, [-1, 1, 1, 1]))
        z = tf.concat([z1, z2], 3)
        return z
