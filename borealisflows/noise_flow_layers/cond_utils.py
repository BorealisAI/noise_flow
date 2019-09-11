# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import tensorflow as tf


def sdn_iso_model_params_3(iso):
    init_val = - 6.0
    p1 = tf.get_variable("p1", [1], tf.float32, initializer=tf.constant_initializer(init_val))
    p2 = tf.get_variable("p2", [1], tf.float32, initializer=tf.constant_initializer(init_val))
    p3 = tf.get_variable("p3", [1], tf.float32, initializer=tf.constant_initializer(init_val))
    beta1 = tf.exp(p1) * iso[0] ** 2 + tf.exp(p2) * iso[0] + tf.exp(p3)

    q1 = tf.get_variable("q1", [1], tf.float32, initializer=tf.constant_initializer(init_val))
    q2 = tf.get_variable("q2", [1], tf.float32, initializer=tf.constant_initializer(init_val))
    q3 = tf.get_variable("q3", [1], tf.float32, initializer=tf.constant_initializer(init_val))
    q4 = tf.get_variable("q4", [1], tf.float32, initializer=tf.constant_initializer(init_val))
    beta2 = tf.exp(q1) * iso[0] ** 3 + tf.exp(q2) * iso[0] ** 2 + tf.exp(q3) * iso[0] + tf.exp(q4)

    return beta1, beta2


def sdn_iso_model_params_2(iso):
    init_val = - 6.0
    p2 = tf.get_variable("p2", [1], tf.float32, initializer=tf.constant_initializer(init_val))
    p3 = tf.get_variable("p3", [1], tf.float32, initializer=tf.constant_initializer(init_val))
    beta1 = tf.exp(p2) * iso[0] + tf.exp(p3)

    q2 = tf.get_variable("q2", [1], tf.float32, initializer=tf.constant_initializer(init_val))
    q3 = tf.get_variable("q3", [1], tf.float32, initializer=tf.constant_initializer(init_val))
    q4 = tf.get_variable("q4", [1], tf.float32, initializer=tf.constant_initializer(init_val))
    beta2 = tf.exp(q2) * iso[0] ** 2 + tf.exp(q3) * iso[0] + tf.exp(q4)

    return beta1, beta2


def sdn_model_params(yy):
    # init_val = - 6.0
    b1 = tf.get_variable("b1", [1], tf.float32, initializer=tf.constant_initializer(-3.0))  # sigmoid(-3) --> 0
    b1 = tf.nn.sigmoid(b1, 'sigmoid_b1')
    b2 = tf.get_variable("b2", [1], tf.float32, initializer=tf.constant_initializer(3.0))  # sigmoid(3) --> 1
    b2 = tf.nn.sigmoid(b2, 'sigmoid_b2')
    scale = tf.sqrt(b1 * yy + b2)

    # tf.summary.histogram('sdn/b1', b1)
    # tf.summary.histogram('sdn/b2', b2)

    return scale


def sdn_model_params_ex1(yy, iso):
    # c = 1e-5
    # c = 1
    c = 1e-2
    # init = -5.0
    init = 0.0

    rg00100 = tf.get_variable("r_gain_param_%05d" % 100, [1], tf.float32, initializer=tf.constant_initializer(init / c))
    rg00400 = tf.get_variable("r_gain_param_%05d" % 400, [1], tf.float32, initializer=tf.constant_initializer(init / c))
    rg00800 = tf.get_variable("r_gain_param_%05d" % 800, [1], tf.float32, initializer=tf.constant_initializer(init / c))
    rg01600 = tf.get_variable("r_gain_param_%05d" % 1600, [1], tf.float32,
                              initializer=tf.constant_initializer(init / c))
    rg03200 = tf.get_variable("r_gain_param_%05d" % 3200, [1], tf.float32,
                              initializer=tf.constant_initializer(init / c))

    r_gain = tf.cond(tf.equal(iso[0], 100),
                     lambda: tf.exp(c * rg00100) * iso,
                     lambda: tf.cond(tf.equal(iso[0], 400),
                                     lambda: tf.exp(c * rg00400) * iso,
                                     lambda: tf.cond(tf.equal(iso[0], 800),
                                                     lambda: tf.exp(c * rg00800) * iso,
                                                     lambda: tf.cond(tf.equal(iso[0], 1600),
                                                                     lambda: tf.exp(c * rg01600) * iso,
                                                                     lambda: tf.cond(tf.equal(iso[0], 3200),
                                                                                     lambda: tf.exp(c * rg03200)
                                                                                             * iso,
                                                                                     lambda: tf.exp(c * rg00800)
                                                                                             * iso
                                                                                     )
                                                                     )
                                                     )
                                     )
                     )

    b1 = tf.get_variable("b1", [1], tf.float32, initializer=tf.constant_initializer(-3.0))  # sigmoid(-3) --> 0
    b1 = tf.nn.sigmoid(b1, 'sigmoid_b1')
    b2 = tf.get_variable("b2", [1], tf.float32, initializer=tf.constant_initializer(3.0))  # sigmoid(3) --> 1
    b2 = tf.nn.sigmoid(b2, 'sigmoid_b2')
    scale = tf.sqrt(b1 * yy / r_gain + b2)

    # tf.summary.histogram('sdn/b1', b1)
    # tf.summary.histogram('sdn/b2', b2)

    return scale


def sdn_model_params_ex2(yy, iso, gain_init):
    c = 1e-1
    g00100 = tf.get_variable("gain_param_%05d" % 100, [1], tf.float32,
                             initializer=tf.constant_initializer(gain_init / c))
    g00400 = tf.get_variable("gain_param_%05d" % 400, [1], tf.float32,
                             initializer=tf.constant_initializer(gain_init / c))
    g00800 = tf.get_variable("gain_param_%05d" % 800, [1], tf.float32,
                             initializer=tf.constant_initializer(gain_init / c))
    g01600 = tf.get_variable("gain_param_%05d" % 1600, [1], tf.float32,
                             initializer=tf.constant_initializer(gain_init / c))
    g03200 = tf.get_variable("gain_param_%05d" % 3200, [1], tf.float32,
                             initializer=tf.constant_initializer(gain_init / c))
    gain = tf.cond(tf.equal(iso[0], 100),
                   lambda: tf.exp(c * g00100) * iso,
                   lambda: tf.cond(tf.equal(iso[0], 400),
                                   lambda: tf.exp(c * g00400) * iso,
                                   lambda: tf.cond(tf.equal(iso[0], 800),
                                                   lambda: tf.exp(c * g00800) * iso,
                                                   lambda: tf.cond(tf.equal(iso[0], 1600),
                                                                   lambda: tf.exp(c * g01600) * iso,
                                                                   lambda: tf.cond(tf.equal(iso[0], 3200),
                                                                                   lambda: tf.exp(c * g03200) * iso,
                                                                                   lambda: tf.exp(c * g00800) * iso
                                                                                   )
                                                                   )
                                                   )
                                   )
                   )
    b1 = tf.get_variable("b1", [1], tf.float32, initializer=tf.constant_initializer(-3.0))  # sigmoid(-3) --> 0
    b1 = tf.nn.sigmoid(b1, 'sigmoid_b1')
    b2 = tf.get_variable("b2", [1], tf.float32, initializer=tf.constant_initializer(3.0))  # sigmoid(3) --> 1
    b2 = tf.nn.sigmoid(b2, 'sigmoid_b2')
    scale = tf.sqrt(gain * (b1 * yy / gain + b2))

    # tf.summary.histogram('sdn/b1', b1)
    # tf.summary.histogram('sdn/b2', b2)

    return scale


def sdn_model_params_ex3(yy, iso, gain_init):
    c = 1e-1
    g00100 = tf.get_variable("gain_param_%05d" % 100, [1], tf.float32,
                             initializer=tf.constant_initializer(gain_init / c))
    g00400 = tf.get_variable("gain_param_%05d" % 400, [1], tf.float32,
                             initializer=tf.constant_initializer(gain_init / c))
    g00800 = tf.get_variable("gain_param_%05d" % 800, [1], tf.float32,
                             initializer=tf.constant_initializer(gain_init / c))
    g01600 = tf.get_variable("gain_param_%05d" % 1600, [1], tf.float32,
                             initializer=tf.constant_initializer(gain_init / c))
    g03200 = tf.get_variable("gain_param_%05d" % 3200, [1], tf.float32,
                             initializer=tf.constant_initializer(gain_init / c))
    gain = tf.cond(tf.equal(iso[0], 100),
                   lambda: tf.exp(c * g00100) * iso,
                   lambda: tf.cond(tf.equal(iso[0], 400),
                                   lambda: tf.exp(c * g00400) * iso,
                                   lambda: tf.cond(tf.equal(iso[0], 800),
                                                   lambda: tf.exp(c * g00800) * iso,
                                                   lambda: tf.cond(tf.equal(iso[0], 1600),
                                                                   lambda: tf.exp(c * g01600) * iso,
                                                                   lambda: tf.cond(tf.equal(iso[0], 3200),
                                                                                   lambda: tf.exp(c * g03200) * iso,
                                                                                   lambda: tf.exp(c * g00800) * iso
                                                                                   )
                                                                   )
                                                   )
                                   )
                   )
    b1 = tf.get_variable("b1", [1], tf.float32, initializer=tf.constant_initializer(-3.0))  # sigmoid(-3) --> 0
    b1 = tf.nn.sigmoid(b1, 'sigmoid_b1')
    b2 = tf.get_variable("b2", [1], tf.float32, initializer=tf.constant_initializer(3.0))  # sigmoid(3) --> 1
    b2 = tf.nn.sigmoid(b2, 'sigmoid_b2')
    # scale = tf.sqrt(gain * (b1 * yy / gain + b2))
    scale = gain * tf.sqrt(b1 * yy / gain + b2)
    return scale


def sdn_model_params_ex4(yy, iso, gain_init):
    # c = 1e-1
    c = 1
    with tf.variable_scope('sdn_gain', reuse=tf.AUTO_REUSE):
        # gain = tf.get_variable('gain_val', [1], tf.float32, initializer=tf.constant_initializer(gain_init / c))
        gain = tf.get_variable('gain_val', [1], tf.float32, initializer=tf.constant_initializer(1.0))
        iso_vals = tf.constant([100, 400, 800, 1600, 3200], dtype=tf.float32)
        gain_params = tf.get_variable('gain_params', [iso_vals.shape[0]], tf.float32,
                                      initializer=tf.constant_initializer(gain_init / c))
        # iso_idx = iso_vals.index(iso[0])
        iso_idx = tf.where(tf.equal(iso_vals, iso))
        gain_one_hot = tf.one_hot(iso_idx, iso_vals.shape[0])
        g = tf.reduce_sum(gain_one_hot * gain_params)
        gain = tf.exp(c * g) * iso

        beta1 = tf.get_variable("beta1", [1], tf.float32, initializer=tf.constant_initializer(gain_init / c))
        # beta1 = tf.get_variable("beta1", [1], tf.float32, initializer=tf.constant_initializer(0.0))
        beta2 = tf.get_variable("beta2", [1], tf.float32, initializer=tf.constant_initializer(0.0))
        # b1 = tf.nn.sigmoid(b1, 'sigmoid_b1')
        # b2 = tf.nn.sigmoid(b2, 'sigmoid_b2')
        # scale = tf.sqrt(b1 * yy / gain + b2)
        beta1 = tf.exp(c * beta1)
        beta2 = tf.exp(c * beta2)
        scale = tf.sqrt(beta1 * yy / gain + beta2)
    return scale


def sdn_model_params_ex5(yy, iso, gain_init, cam, param_inits):
    # c = 1e-1
    with tf.variable_scope('sdn_gain', reuse=tf.AUTO_REUSE):
        # params initializers
        (c_i, beta1_i, beta2_i, gain_params_i, cam_params_i) = param_inits

        # cam params
        n_param_per_cam = 3  # for scaling beta1, beta2, and gain
        cam_vals = tf.constant([0, 1, 2, 3, 4], dtype=tf.float32)  # 'IP', 'GP', 'S6', 'N6', 'G4'
        cam_params = tf.get_variable('cam_params', [n_param_per_cam, cam_vals.shape[0]], tf.float32,
                                     initializer=tf.constant_initializer(cam_params_i))  # 1.0
        cam_idx = tf.where(tf.equal(cam_vals, cam))
        cam_idx = cam_idx[0]
        cam_one_hot = tf.one_hot(cam_idx, cam_vals.shape[0])
        one_cam_params = tf.reduce_sum(cam_one_hot * cam_params, axis=1)
        one_cam_params = tf.exp(c_i * one_cam_params)

        # gain params
        gain = tf.get_variable('gain_val', [1], tf.float32, initializer=tf.constant_initializer(1.0))
        iso_vals = tf.constant([100, 400, 800, 1600, 3200], dtype=tf.float32)
        gain_params = tf.get_variable('gain_params', [iso_vals.shape[0]], tf.float32,
                                      initializer=tf.constant_initializer(gain_params_i))  # -5.0 / c_i
        iso_idx = tf.where(tf.equal(iso_vals, iso))
        gain_one_hot = tf.one_hot(iso_idx, iso_vals.shape[0])
        g = tf.reduce_sum(gain_one_hot * gain_params)
        gain = tf.exp(c_i * g * one_cam_params[2]) * iso

        beta1 = tf.get_variable("beta1", [1], tf.float32,
                                initializer=tf.constant_initializer(beta1_i))  # -5.0 / c_i
        beta2 = tf.get_variable("beta2", [1], tf.float32,
                                initializer=tf.constant_initializer(beta2_i))  # 0.0
        beta1 = tf.exp(c_i * beta1 * one_cam_params[0])
        beta2 = tf.exp(c_i * beta2 * one_cam_params[1])
        scale = tf.sqrt(beta1 * yy / gain + beta2)
    return scale


def sdn_model_params_ex6(yy, iso, gain_init, cam, param_inits):
    # c = 1e-1
    with tf.variable_scope('sdn_gain', reuse=tf.AUTO_REUSE):
        # params initializers
        (c_i, beta1_i, beta2_i, gain_params_i, cam_params_i) = param_inits

        # cam params
        n_param_per_cam = 1  # for scaling beta1, beta2, and gain
        cam_vals = tf.constant([0, 1, 2, 3, 4], dtype=tf.float32)  # 'IP', 'GP', 'S6', 'N6', 'G4'
        cam_params = tf.get_variable('cam_params', [n_param_per_cam, cam_vals.shape[0]], tf.float32,
                                     initializer=tf.constant_initializer(cam_params_i))  # 1.0
        cam_idx = tf.where(tf.equal(cam_vals, cam))
        cam_idx = cam_idx[0]
        cam_one_hot = tf.one_hot(cam_idx, cam_vals.shape[0])
        one_cam_params = tf.reduce_sum(cam_one_hot * cam_params, axis=1)
        one_cam_params = tf.exp(c_i * one_cam_params)

        # gain params
        gain = tf.get_variable('gain_val', [1], tf.float32, initializer=tf.constant_initializer(1.0))
        iso_vals = tf.constant([100, 400, 800, 1600, 3200], dtype=tf.float32)
        gain_params = tf.get_variable('gain_params', [iso_vals.shape[0]], tf.float32,
                                      initializer=tf.constant_initializer(gain_params_i))  # -5.0 / c_i
        iso_idx = tf.where(tf.equal(iso_vals, iso))
        gain_one_hot = tf.one_hot(iso_idx, iso_vals.shape[0])
        g = tf.reduce_sum(gain_one_hot * gain_params)
        gain = tf.exp(c_i * g * one_cam_params[0]) * iso

        beta1 = tf.get_variable("beta1", [1], tf.float32,
                                initializer=tf.constant_initializer(beta1_i))  # -5.0 / c_i
        beta2 = tf.get_variable("beta2", [1], tf.float32,
                                initializer=tf.constant_initializer(beta2_i))  # 0.0
        beta1 = tf.exp(c_i * beta1)
        beta2 = tf.exp(c_i * beta2)
        scale = tf.sqrt(beta1 * yy / gain + beta2)
    return scale


def sdn_model_params_ex7(yy, iso, gain_init, cam, param_inits):  # TODO
    # c = 1e-1
    with tf.variable_scope('sdn_gain', reuse=tf.AUTO_REUSE):
        # params initializers
        (c_i, beta1_i, beta2_i, gain_params_i, cam_params_i) = param_inits

        # cam params
        n_param_per_cam = 1  # for scaling beta1, beta2, and gain
        cam_vals = tf.constant([0, 1, 2, 3, 4], dtype=tf.float32)  # 'IP', 'GP', 'S6', 'N6', 'G4'
        cam_params = tf.get_variable('cam_params', [n_param_per_cam, cam_vals.shape[0]], tf.float32,
                                     initializer=tf.constant_initializer(cam_params_i))  # 1.0
        cam_idx = tf.where(tf.equal(cam_vals, cam))
        cam_idx = cam_idx[0]
        cam_one_hot = tf.one_hot(cam_idx, cam_vals.shape[0])
        one_cam_params = tf.reduce_sum(cam_one_hot * cam_params, axis=1)
        one_cam_params = tf.exp(c_i * one_cam_params)

        # gain params
        gain = tf.get_variable('gain_val', [1], tf.float32, initializer=tf.constant_initializer(1.0))
        # iso_vals = tf.constant([100, 400, 800, 1600, 3200], dtype=tf.float32)
        # gain_params = tf.get_variable('gain_params', [iso_vals.shape[0]], tf.float32,
        #                               initializer=tf.constant_initializer(gain_params_i))  # -5.0 / c_i
        # iso_idx = tf.where(tf.equal(iso_vals, iso))
        # gain_one_hot = tf.one_hot(iso_idx, iso_vals.shape[0])
        # g = tf.reduce_sum(gain_one_hot * gain_params)
        g0 = tf.get_variable("g0", [1], tf.float32, initializer=tf.constant_initializer(0.0))
        g1 = tf.get_variable("g1", [1], tf.float32, initializer=tf.constant_initializer(0.0))
        g2 = tf.get_variable("g2", [1], tf.float32, initializer=tf.constant_initializer(0.0))
        gain = tf.exp(c_i * (g2 * iso * iso + g1 * iso + g0) * one_cam_params[0])

        beta1 = tf.get_variable("beta1", [1], tf.float32,
                                initializer=tf.constant_initializer(beta1_i))  # -5.0 / c_i
        beta2 = tf.get_variable("beta2", [1], tf.float32,
                                initializer=tf.constant_initializer(beta2_i))  # 0.0
        beta1 = tf.exp(c_i * beta1)
        beta2 = tf.exp(c_i * beta2)
        scale = tf.sqrt(beta1 * yy / gain + beta2)
    return scale


def gain_model_params(gain):
    # init_val = 1e-2
    g1 = tf.get_variable("g1", [1], tf.float32, initializer=tf.constant_initializer(-3.0))
    g1 = tf.nn.sigmoid(g1, 'sigmoid_g1')
    g2 = tf.get_variable("g2", [1], tf.float32, initializer=tf.constant_initializer(3.0))
    g2 = tf.nn.sigmoid(g2, 'sigmoid_g2')
    scale = g1 * gain + g2

    # tf.summary.histogram('gain/g1', g1)
    # tf.summary.histogram('gain/g2', g2)

    return scale


def gain_model_params_ex1(gain):
    c = 1e-5
    # c = 1
    # c = 1e-2

    # g1 = tf.get_variable("g1", [1], tf.float32, initializer=tf.constant_initializer( 0.0/c))
    # g2 = tf.get_variable("g2", [1], tf.float32, initializer=tf.constant_initializer(-5.0/c))

    g1 = tf.get_variable("g1", [1], tf.float32, initializer=tf.constant_initializer(-5.0 / c))
    g2 = tf.get_variable("g2", [1], tf.float32, initializer=tf.constant_initializer(0.0 / c))

    scale = tf.exp(c * g1) * gain + tf.exp(c * g2)

    # tf.summary.histogram('gain/c1', c1)
    # tf.summary.histogram('gain/g1', g1)
    # tf.summary.histogram('gain/g2', g2)

    return scale


def gain_model_params_ex2(iso, gain_init):
    """Using different variable for each ISO"""
    # c = 1e-5
    # c = 1
    # c = 1e-2
    c = 1e-1
    # init = -5.0
    init = gain_init
    g00100 = tf.get_variable("gain_param_%05d" % 100, [1], tf.float32, initializer=tf.constant_initializer(init / c))
    g00400 = tf.get_variable("gain_param_%05d" % 400, [1], tf.float32, initializer=tf.constant_initializer(init / c))
    g00800 = tf.get_variable("gain_param_%05d" % 800, [1], tf.float32, initializer=tf.constant_initializer(init / c))
    g01600 = tf.get_variable("gain_param_%05d" % 1600, [1], tf.float32, initializer=tf.constant_initializer(init / c))
    g03200 = tf.get_variable("gain_param_%05d" % 3200, [1], tf.float32, initializer=tf.constant_initializer(init / c))

    scale = tf.cond(tf.equal(iso[0], 100),
                    lambda: tf.exp(c * g00100) * iso,
                    lambda: tf.cond(tf.equal(iso[0], 400),
                                    lambda: tf.exp(c * g00400) * iso,
                                    lambda: tf.cond(tf.equal(iso[0], 800),
                                                    lambda: tf.exp(c * g00800) * iso,
                                                    lambda: tf.cond(tf.equal(iso[0], 1600),
                                                                    lambda: tf.exp(c * g01600) * iso,
                                                                    lambda: tf.cond(tf.equal(iso[0], 3200),
                                                                                    lambda: tf.exp(c * g03200) * iso,
                                                                                    lambda: tf.exp(c * g00800) * iso
                                                                                    )
                                                                    )
                                                    )
                                    )
                    )

    # scale = tf.exp(c * g00100) * iso

    # tf.summary.histogram('gain_param_%05d' % 100, g00100)
    # tf.summary.histogram('gain_param_%05d' % 400, g00400)
    # tf.summary.histogram('gain_param_%05d' % 800, g00800)
    # tf.summary.histogram('gain_param_%05d' % 1600, g01600)
    # tf.summary.histogram('gain_param_%05d' % 3200, g03200)

    return scale


def gain_model_params_ex3(iso):
    """Using different variable for each ISO"""
    c = 1e-5
    # c = 1
    # c = 1e-2

    g00100 = tf.get_variable("gain_param_%05d" % 100, [1], tf.float32, initializer=tf.constant_initializer(-5.0 / c))
    g00400 = tf.get_variable("gain_param_%05d" % 400, [1], tf.float32, initializer=tf.constant_initializer(-5.0 / c))
    g00800 = tf.get_variable("gain_param_%05d" % 800, [1], tf.float32, initializer=tf.constant_initializer(-5.0 / c))
    g01600 = tf.get_variable("gain_param_%05d" % 1600, [1], tf.float32, initializer=tf.constant_initializer(-5.0 / c))
    g03200 = tf.get_variable("gain_param_%05d" % 3200, [1], tf.float32, initializer=tf.constant_initializer(-5.0 / c))

    scale = tf.cond(tf.equal(iso[0], 100),
                    lambda: tf.exp(c * g00100),
                    lambda: tf.cond(tf.equal(iso[0], 400),
                                    lambda: tf.exp(c * g00400),
                                    lambda: tf.cond(tf.equal(iso[0], 800),
                                                    lambda: tf.exp(c * g00800),
                                                    lambda: tf.cond(tf.equal(iso[0], 1600),
                                                                    lambda: tf.exp(c * g01600),
                                                                    lambda: tf.cond(tf.equal(iso[0], 3200),
                                                                                    lambda: tf.exp(c * g03200),
                                                                                    lambda: tf.exp(c * g00800)
                                                                                    )
                                                                    )
                                                    )
                                    )
                    )
    # scale = tf.exp(c * g00100) * iso
    # tf.summary.histogram('gain_param_%05d' % 100, g00100)
    # tf.summary.histogram('gain_param_%05d' % 400, g00400)
    # tf.summary.histogram('gain_param_%05d' % 800, g00800)
    # tf.summary.histogram('gain_param_%05d' % 1600, g01600)
    # tf.summary.histogram('gain_param_%05d' % 3200, g03200)
    return scale


def gain_model_params_ex4(iso, gain_init):
    """Using different variable for each ISO"""
    # c = 1e-1
    # c = 1
    # init = gain_init
    with tf.variable_scope('sdn_gain', reuse=tf.AUTO_REUSE):
        scale = tf.get_variable('gain_val', [1], tf.float32, initializer=tf.constant_initializer(1.0))
        # scale = tf.get_variable('gain_val')
    return scale
