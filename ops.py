import tensorflow as tf


def dense(out_dim, input, activation=tf.nn.relu):
    in_dim = tf.shape(input)[1]
    W = tf.Variable(tf.random_normal([in_dim, out_dim]))
    b = tf.Variable(tf.zeros([out_dim]))

    output = tf.matmul(input, W) + b
    if activation:
        output = activation(output)

    return output
