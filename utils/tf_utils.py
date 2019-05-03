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

def build_mlp(
    input_placeholder,
    output_size,
    scope,
    n_layers=1,
    size=256,
    activation=tf.nn.relu,
    output_activation=None,
    reuse=False
      ):
    out = tf.cast(input_placeholder, tf.float32)
    with tf.variable_scope(scope, reuse=reuse):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out