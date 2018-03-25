import tensorflow as tf


def weights(shape, initializer='variance_scaling', reg_norm=True, reg_coef=0.001, name='variable'):
    if initializer == 'variance_scaling':
        init = tf.variance_scaling_initializer()
    else:
        init = tf.glorot_normal_initializer()
    weight = tf.Variable(init(shape), name=name)

    if reg_norm:
        l2_norm = reg_coef * tf.reduce_sum(tf.squared(weight))
        return weight, l2_norm

    return weight

def dense(out_dim, input, mean=0, var=1., activation=tf.nn.relu):
    in_dim = input.get_shape()[-1].value
    W = tf.Variable(tf.random_normal([in_dim, out_dim], mean, var))
    b = tf.Variable(tf.zeros([out_dim]))

    output = tf.matmul(input, W) + b
    if activation:
        output = activation(output)

    return output

def batch_norm(x, if_train, type='conv', epsilon=1e-3, decay=0.9):
    '''
    Batch normalization layer
    Args:
        - type: either `conv` or `fc`; determines which axes
                to aggregate moments over.
    '''
    if type == 'conv':
        axes = [0, 1, 2]
    else:
        axes = [0]

    '''
    During training, we use the mean and variance per batch
    to standardize; however, during evaluation, we want to use
    an exponential moving average of the batch means we've
    seen thus far.
    '''
    mean, var = tf.nn.moments(x, axes, keep_dims=True)
    ema = tf.train.ExponentialMovingAverage(decay=decay)
    ema_op = ema.apply([mean, var])

    with tf.control_dependencies([ema_op]):
        ema_mean, ema_var = tf.identity(mean), tf.identity(var)

    batch_mean, batch_var = tf.cond(
        if_train,
        lambda : (ema_mean, ema_var),
        lambda : (ema.average(mean), ema.average(var)))
    
    out_shape = tf.shape(mean)
    gamma = tf.Variable(
        tf.constant(1., shape=out_shape),
        name='gamma',
        trainable=True)
    beta = tf.Variable(
        tf.constant(0., shape=out_shape),
        name='beta',
        trainable=True)

    # Apply the tensorflow batch normalization operation
    y = tf.nn.batch_normalization(x, batch_mean,
        batch_var, beta, gamma, epsilon)

    return y
