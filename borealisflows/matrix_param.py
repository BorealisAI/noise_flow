# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import scipy as sp
import tensorflow as tf
import tensorflow.contrib.distributions as tfdist
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import array_ops

DTYPE = tf.float32

__all__ = ["matrix_param",
           "matrix_param_lu",
           "matrix_param_none"]


def matrix_param_none(name, init_A, dtype=None):
    A = tf.get_variable("A_matpar_none_{}".format(name),
                        dtype=dtype, initializer=init_A)
    A_inv = tf.matrix_inverse(A)
    _, log_abs_det = tf.linalg.slogdet(A)

    return {'A': A, 'A_inv': A_inv, 'log_abs_det': log_abs_det}

def _vec2stricttri(input, upper, name=None):  # pylint: disable=redefined-builtin
    with ops.name_scope(name, 'vec2stricttri', [input]):
        matrix = ops.convert_to_tensor(input, name='input')
        if matrix.shape[-1:] == [0]:
            return matrix
        batch_shape = matrix.shape[:-1]
        if not batch_shape.is_fully_defined():
            batch_shape = array_ops.shape(matrix)[:-1]

        matrix = array_ops.reshape(
            matrix, array_ops.concat(([-1], array_ops.shape(matrix)[-1:]), axis=0))

        # build a triangular matrix
        M_tri_base = tfdist.fill_triangular(matrix, upper=upper)

        # make it strictly triangular
        if upper:
            result = array_ops.pad(M_tri_base, [[0, 0], [0, 1], [1, 0]])
        else:
            result = array_ops.pad(M_tri_base, [[0, 0], [1, 0], [0, 1]])

        if not matrix.shape.is_fully_defined():
            return array_ops.reshape(
                result,
                array_ops.concat((batch_shape, array_ops.shape(result)[-2:]), axis=0))
        return array_ops.reshape(result, batch_shape.concatenate(result.shape[-2:]))


def _stricttri2vec(input, upper, name=None):  # pylint: disable=redefined-builtin
    with ops.name_scope(name, 'stricttri2vec', [input]):
        matrix = ops.convert_to_tensor(input, name='input')
        if matrix.shape[-2:] == [0, 0]:
            return matrix
        batch_shape = matrix.shape[:-2]
        if not batch_shape.is_fully_defined():
            batch_shape = array_ops.shape(matrix)[:-2]

        matrix = array_ops.reshape(
            matrix, array_ops.concat(([-1], array_ops.shape(matrix)[-2:]), axis=0))

        # only keep triangular part
        if upper:
            # remove last row and first column
            matrix_trim = matrix[:, :-1, 1:]
            # upper triangular part
            matrix_tri = gen_array_ops.matrix_band_part(matrix_trim, 0, -1)
        else:
            # remove first row and last column
            matrix_trim = matrix[:, 1:, :-1]
            # lower triangular part (including the diagonal)
            matrix_tri = gen_array_ops.matrix_band_part(matrix_trim, -1, 0)

        if matrix_trim.shape[-2:] == [0, 0]:
            return matrix_trim

        # vectorize the triangular portion
        result = tfdist.fill_triangular_inverse(matrix_tri, upper=upper)

        if not matrix.shape.is_fully_defined():
            ret = array_ops.reshape(
                result,
                array_ops.concat((batch_shape, array_ops.shape(result)[-1:]), axis=0))
        else:
            ret = array_ops.reshape(
                result, batch_shape.concatenate(result.shape[-1:]))

        return ret


def matrix_param_lu(name, init_A, dtype=None):
    A_shape = init_A.shape
    np_p, np_l, np_u = sp.linalg.lu(init_A)

    np_s = np.diag(np_u)
    np_sign_s = np.sign(np_s)
    np_log_s = np.log(abs(np_s))
    np_u = np.triu(np_u, k=1)

    p = tf.get_variable("P_matpar_lu_{}".format(
        name), initializer=np_p, dtype=dtype, trainable=False)
    sign_s = tf.get_variable("sign_S_matpar_lu_{}".format(
        name), initializer=np_sign_s, dtype=dtype, trainable=False)

    log_s = tf.get_variable("log_S_matpar_lu_{}".format(
        name), initializer=np_log_s, dtype=dtype)

    init_l_vec = _stricttri2vec(np_l, upper=False)
    l_vec = tf.get_variable("L_vec_matpar_lu_{}".format(
        name), initializer=init_l_vec, dtype=dtype)
    l_base = _vec2stricttri(l_vec, upper=False)
    l = tf.matrix_set_diag(l_base, tf.ones_like(log_s))
    # l =  tf.Print(l)

    init_u_vec = _stricttri2vec(np_u, upper=True)
    u_vec = tf.get_variable("U_vec_matpar_lu_{}".format(
        name), initializer=init_u_vec, dtype=dtype)
    u_base = _vec2stricttri(u_vec, upper=True)
    u = tf.matrix_set_diag(u_base, sign_s * tf.exp(log_s))

    A = tf.matmul(p, tf.matmul(l, u))

    # Inverse of a permutation matrix is just its transpose
    p_inv = tf.transpose(p)
    # A_inv = U_inv * L_inv * P_inv
    A_inv = tf.linalg.triangular_solve(
        u, tf.linalg.triangular_solve(l, p_inv, lower=True), lower=False)

    log_abs_det = tf.reduce_sum(log_s)

    return {'A': A, 'A_inv': A_inv, 'log_abs_det': log_abs_det}


def matrix_param_lu2(name, init_A, dtype=None):
    A_shape = init_A.shape
    np_p, np_l, np_u = sp.linalg.lu(init_A)
    np_s = np.diag(np_u)
    np_sign_s = np.sign(np_s)
    np_log_s = np.log(abs(np_s))
    np_u = np.triu(np_u, k=1)

    p = tf.get_variable(
        "P_{}".format(name), initializer=np_p, trainable=False)
    l = tf.get_variable(
        "L_filters_{}".format(name), initializer=np_l)
    sign_s = tf.get_variable(
        "sign_S_{}".format(name),
        initializer=np_sign_s, trainable=False)
    log_s = tf.get_variable(
        "log_S_filters_{}".format(name), initializer=np_log_s)
    u = tf.get_variable(
        "U_filters_{}".format(name), initializer=np_u)

    dtype2 = 'float64'
    p = tf.cast(p, dtype2)
    l = tf.cast(l, dtype2)
    sign_s = tf.cast(sign_s, dtype2)
    log_s = tf.cast(log_s, dtype2)
    u = tf.cast(u, dtype2)

    l_mask = np.tril(np.ones(A_shape, dtype=dtype2), -1)
    l = l * l_mask + tf.eye(*A_shape, dtype=dtype2)
    u = u * np.transpose(l_mask) + tf.diag(sign_s * tf.exp(log_s))
    A = tf.matmul(p, tf.matmul(l, u))

    if True:
        u_inv = tf.matrix_inverse(u)
        l_inv = tf.matrix_inverse(l)
        p_inv = tf.matrix_inverse(p)
        A_inv = tf.matmul(u_inv, tf.matmul(l_inv, p_inv))
    else:
        A_inv = tf.matrix_inverse(A)

    A = tf.cast(A, dtype)
    A_inv = tf.cast(A_inv, dtype)
    log_s = tf.cast(log_s, dtype)

    log_abs_det = tf.reduce_sum(log_s)
    return {'A': A, 'A_inv': A_inv, 'log_abs_det': log_abs_det}


_params = {'LU': matrix_param_lu,
           'LU2': matrix_param_lu2,
           'NONE': matrix_param_none}


def matrix_param(param_type, name, init_A, dtype=None):
    A_shape = init_A.shape
    if (A_shape == (0, 0) or A_shape == (1, 1)) and param_type != 'NONE':
        print('\n\nWARNING: requested a non-trivial parameterization of a 1x1 matrix, ignoring\n\n')
        # FIXME: do the sign/log paramaterization here instead.  Seems to work better
        # TODO: Also look into the fact that this code path is getting hit regularly...
        return _params['NONE'](name, init_A, dtype)
    else:
        return _params[param_type](name, init_A, dtype)
