import tensorflow as tf
import os

def save_tf_vars(sess, scope, path):
    saver = tf.train.Saver([v for v in tf.global_variables() if v.name.startswith(scope + '/')])
    if not os.path.exists(path):
        os.makedirs(path)
    saver.save(sess, save_path=path)


def load_tf_vars(sess, scope, path):
    saver = tf.train.Saver([v for v in tf.global_variables() if v.name.startswith(scope + '/')])
    saver.restore(sess, path)


def os_setup(gpu_num=2):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)


# Adapted from https://github.com/zoli333/Weight-Normalization
def dense_weight_norm(x, num_units, nonlinearity=None, init_scale=1., init=False, name=''):
    with tf.variable_scope(name):
        V = tf.get_variable('V', shape=[int(x.get_shape()[1]), num_units], dtype=tf.float32,
                            initializer=tf.random_normal_initializer(0, 0.05), trainable=True)

        b = tf.get_variable('b', shape=[num_units], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.), trainable=True)

        g = tf.get_variable('g', shape=[num_units], dtype=tf.float32,
                            initializer=tf.constant_initializer(1.), trainable=True)
        if init:
            v_norm = tf.nn.l2_normalize(V, [0])
            x = tf.matmul(x, v_norm)
            m_init, v_init = tf.nn.moments(x, [0])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            # FIXME created but never run
            g = g.assign(scale_init)
            b = b.assign(-m_init * scale_init)
            x = tf.reshape(scale_init, [1, num_units]) * (x - tf.reshape(m_init, [1, num_units]))
        else:
            x = tf.matmul(x, V)
            scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
            x = tf.reshape(scaler, [1, num_units]) * x
            b = tf.reshape(b, [1, num_units])
            x = x + b

        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)

        return x


def build_mlp(
    input_placeholder,
    output_size,
    scope,
    n_layers=1,
    size=256,
    activation=tf.nn.relu,
    output_activation=None,
    layer_norm=False,
    weight_norm=True,
    reuse=False):

    out = tf.cast(input_placeholder, tf.float32)
    with tf.variable_scope(scope, reuse=reuse):
        for _ in range(n_layers):
            if weight_norm:
                out = dense_weight_norm(out, size, nonlinearity=activation)
            else:
                out = tf.layers.dense(out, size, activation=activation)
            if layer_norm:
                out = tf.contrib.layers.layer_norm(out)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out
