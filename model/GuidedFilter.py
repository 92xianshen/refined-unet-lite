import numpy as np
import tensorflow as tf

class GuidedFilter(tf.keras.layers.Layer):
    """ A layer implementing guided filter """
    
    def __init__(self):
        super(GuidedFilter, self).__init__()

    def call(self, I, p, r, eps=1e-8):
        def diff_x(inputs, r):
            assert inputs.shape.ndims == 4

            left    = inputs[:,         r:2 * r + 1]
            middle  = inputs[:, 2 * r + 1:         ] - inputs[:,           :-2 * r - 1]
            right   = inputs[:,        -1:         ] - inputs[:, -2 * r - 1:    -r - 1]

            outputs = tf.concat([left, middle, right], axis=1)

            return outputs

        def diff_y(inputs, r):
            assert inputs.shape.ndims == 4

            left    = inputs[:, :,         r:2 * r + 1]
            middle  = inputs[:, :, 2 * r + 1:         ] - inputs[:, :,           :-2 * r - 1]
            right   = inputs[:, :,        -1:         ] - inputs[:, :, -2 * r - 1:    -r - 1]

            outputs = tf.concat([left, middle, right], axis=2)

            return outputs

        def box_filter(x, r):
            assert x.shape.ndims == 4

            return diff_y(tf.cumsum(diff_x(tf.cumsum(x, axis=1), r), axis=2), r)

        assert I.shape.ndims == 4 and p.shape.ndims == 4

        I_shape = tf.shape(I)
        p_shape = tf.shape(p)

        # N
        N = box_filter(tf.ones((1, I_shape[1], I_shape[2], 1), dtype=I.dtype), r)

        # mean_x
        mean_I = box_filter(I, r) / N
        # mean_y
        mean_p = box_filter(p, r) / N
        # cov_xy
        cov_Ip = box_filter(I * p, r) / N - mean_I * mean_p
        # var_x
        var_I = box_filter(I * I, r) / N - mean_I * mean_I

        # A
        A = cov_Ip / (var_I + eps)
        # b
        b = mean_p - A * mean_I

        mean_A = box_filter(A, r) / N
        mean_b = box_filter(b, r) / N

        q = mean_A * I + mean_b

        return q