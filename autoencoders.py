import tensorflow as tf

from ops import dense
from utils import Batcher

class ConvVAE:
    '''
    Convolutional Variational Autoencoder
    '''
    def __init__(self, kernel_sizes, layer_sizes, input_spec=None, batch_generator=Batcher):
        self._kernel_sizes = kernel_sizes
        self._layer_sizes = layer_sizes
        self._input_spec = input_spec

        self._batcher = batch_generator
        self._losses_tr = []
        self._losses_val = []

        self._kernels = None
        self._sess = None
        self._built = False

    def _build(self):
        # Inputs
        height, width, depth = self._input_spec
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, height, width, depth])

        # Build the network
        self.encoding = self._build_encoder(self.inputs)
        self.z_sample = self._build_posterior(self.encoding)
        self.decoding = self._build_decoder(self.z_sample)
        vae_loss = self._build_loss(decoding)

        self._built = True

    def _build_input_spec(self, input):
        self._input_spec = input.shape[1:]

    def _build_posterior(self, encoding):
        '''
        Build layers for posterior parameter approximation.
        Args:
            - encoding: a tensor containing the learned representation of
                        model inputs that will be linearly transformed to
                        approximate the parameters to our latent posterior
        Returns:
            - z_sample: a sampled tensor from the approximated distribution
                        conditioned on the input.
        '''
        # Reshape the encoding to shape [batch_size, flattened_dim]
        h, w, d = encoding.get_shape().as_list()[1:]
        flat_dim = h * w * d
        z_dim = self._z_dim
        flat_encoding = tf.reshape(encoding, shape=[-1, flat_dim])

        # Dense layers for approximating the latent mean and log stddev
        # We approximate log sttdev instead of stddev to ensure nonnegativity
        self.z_mu = dense(z_dim, flat_encoding)
        self.z_log_sigma = 0.5 * dense(z_dim, flat_encoding)
        self.z_sigma = tf.exp(self.z_log_sigma)

        # Sample from posterior using the 'reparameterization trick'
        z_sample = self.z_mu + self.z_sigma * tf.random_normal()
        return z_sample

    def _build_encoder(self, inputs):
        '''
        Build convolutional encoder

        Args:
            - inputs: `tf.placeholder` for images
        Returns:
            - layer_inputs: a 4D tensor to be used to estimate posterior parameters
        '''
        kernel_sizes = self._kernel_sizes
        layer_sizes = self._layer_sizes

        kernels = []
        layer_inputs = inputs
        for i in range(len(kernel_sizes)):
            depth_in = layer_inputs.get_shape().as_list()[-1]
            depth_out = layer_sizes[i]

            # Instantiate the sparsely connected shared kernel weights
            W = tf.Variable(
                tf.random_normal(
                    [kernel_sizes[i],
                     kernel_sizes[i],
                     depth_in,
                     depth_out]
                ))
            b = tf.Variable(tf.zeros([depth_out]))
            kernels.append(W)

            # Apply convolution and a relu activation
            layer_inputs = tf.nn.conv2d(input=layer_inputs,
                                        filter=W,
                                        strides=[1, 1, 1, 1],
                                        padding='SAME',
                                        name='enc_conv{}'.format(i))
            layer_inputs = tf.add(layer_inputs, b)
            layer_inputs = tf.nn.relu(layer_inputs)

        self._kernels = kernels

        return layer_inputs

    def _build_decoder(self, z):
        '''
        Deconvolutional decoder; takes a sampled posterior, reshapes
        and deconvolves it using shared encoder weights.
        '''
        kernels = self._kernels
        kernels.reverse()

        # Project and reshape the sampled latent variable so that it can be convolved
        h, w, d = self.encoding.get_shape().as_list()[1:]
        flat_dim = h * w * d
        layer_inputs = dense(flat_dim, z)
        layer_inputs = tf.reshape(layer_inputs, [-1, h, w, d])

        # Encoding convolutions are applied in reverse
        for i in range(len(kernels)):
            depth_out = layer_inputs.get_shape().as_list()[-1]
            W = kernels[i]
            b = tf.Variable(tf.zeros([depth_out]))

            # Deconvolution with shared encoder weights
            layer_inputs = tf.nn.conv2d_transpose(input=layer_inputs,
                                                  filter=W,
                                                  strides=[1, 1, 1, 1],
                                                  padding='SAME',
                                                  name='dec_conv{}'.format(i))
            layer_inputs = tf.add(layer_inputs, b)
            layer_inputs = tf.nn.relu(layer_inputs)

        kernels.reverse()

        return layer_inputs

    def _build_loss(self):
        '''
        Build VAE loss and initialize optimizer.
        '''
        batch_size = tf.shape(self.inputs)[0]
        norm_const = 1./(batch_size * self._input_spec[0] * self._input_spec[1])

        self.recon_loss = norm_const * tf.reduce_sum(tf.squared_difference(self.x, self.x_hat))
        self.kl_loss = (norm_const * -0.5
                        * tf.reduce_sum(1.0
                            - tf.square(self.z_mu)
                            - tf.square(self.z_sigma)
                            + 2.0 * tf.log(self.z_sigma
                                + epsilon)))
        self.loss = self.recon_loss + self.kl_loss
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        return self.loss

    def _init_session(self):
        if self._sess is None:
            self._sess = tf.Session()

        self._sess.run(tf.global_variables_initializer())
        return self._sess

    def _run_train_step(self, inputs, sess=None):
        if sess is None:
            sess = self._init_session()

        loss_, _ = sess.run(
            [self.loss, self.optimizer],
            feed_dict={
                self.inputs: batch
            })
        return loss_

    def _run_homotopy(self, input_instance):
        sess = self._sess
        z_mu, z_sigma = sess.run(
            [self.z_mu, self.z_sigma],
            feed_dict={self.inputs: input_instance})

    def _log(self, loss_val, loss_tr, count):
        print("Iteration {0}: Validation Loss: {1:.4f}"
              "Training Loss: {2:.4f}".format(count, loss_val, loss_tr))

    def _record_metrics(self, loss_tr, loss_val):
        self._losses_tr.append(loss_tr)
        self._losses_val.append(loss_val)

    def fit(self, inputs, val_inputs, n_epochs=10, batch_size=128, val_interval=100):
        if self._input_shape is None:
            self._build_input_spec(inputs)

        if not self._built:
            self._build()

        batcher = self._batcher(inputs, batch_size)
        sess = self._init_session()

        count = 0
        for e in range(n_epochs):
            for batch in batcher.get_batches():
                loss_tr = self._run_train_step(batch, sess)

                count += 1
                if count % val_interval == 0:
                    loss_val, = sess.run(
                        [self.loss],
                        feed_dict={
                            self.inputs: val_inputs
                        })
                    self._log(loss_val, loss_tr, count)
                    self._record_metrics(loss_tr, loss_val)

    def predict(self, inputs):
        sess = self._sess

        reconstructed_input, = sess.run([self.decoding],
            feed_dict={self.inputs: inputs})

        return reconstructed_input
