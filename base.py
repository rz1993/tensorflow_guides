import ops
import tensorflow as tf


class BaseModel:
    def __init__(self, config):
        '''
        Initialization for the BaseModel class.

        :params config: a dictionary of model configurations,
                        which should contain parameter specs and
                        the root directory for model saving and logging.
        '''
        self.config = {k: config[k] for k in config}
        if 'model_dir' not in self.config
            self.config['model_dir'] = 'model_dir'

        # All models should track their own training information
        self.init_train_counters()
        self.init_parameters()

    def add_weight(self,
        shape, trainable=True,
        initializer='variance_scaling',
        reg_norm=True,
        reg_coefficient=0.001,
        loggable=False,
        name='weight'):
        '''
        Add a weight to the model and store it in the
        appropriate parameter collection.
        '''

        tensors = ops.weights(shape,
            initializer=initializer,
            reg_norm=reg_norm,
            reg_coeff=reg_coefficient,
            name=name)
        if reg_norm:
            weight, l2_loss = tensors
        else:
            weight = tensors

        if name in self.params:
            name = name + '_{}'.format(self._w_count)

        self.params[name] = weight
        collection = self.trainable if trainable else non_trainable
        collection[name] = weight
        if reg_norm:
            self.losses[name] = l2_loss
        if loggable:
            self.loggable[name] = weight

        self._w_count += 1

        return weight

    def build(self):
        raise NotImplementedError('Build method must be defined.')

    def init_parameters(self):
        '''
        Initialize a dictionary of trainable and non-trainable
        parameters as well as parameters that should be logged.
        '''
        self._w_count = 0
        self.params = {}
        self.loggable = {}
        self.losses = {}
        self.trainable = {}
        self.non_trainable = {}

    def init_train_counters(self):
        with tf.variable_scope('training_counters'):
            self.curr_epoch = tf.Variable(0,
                trainable=False,
                name='curr_epoch')
            self.global_step = tf.Variable(0,
                trainable=False,
                name='global_step')
            self.incr_epoch = tf.assign(self.curr_epoch, self.curr_epoch + 1)
            self.incr_step = tf.assign(self.global_step, self.global_step + 1)

    def init_saver(self):
        '''
        Save method for the model. Implementation left out because
        weights to be saved must be defined upon initialization.
        '''
        raise NotImplementedError('Save method must be defined')

    def load(self, sess):
        '''
        Load model weights into current active session. Tensors
        for weights must already by defined in the current session.

        :params sess: Tensorflow session object with defined tensors.
        '''
        latest_checkpoint = tf.train.latest_checkpoint(config['model_dir'])
        if latest_checkpoint:
            print("Loading model checkpoint: {}...".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model finished loading.")

    def save(self, sess):
        self.saver.save(sess, self.config['model_dir'], self.global_step)


class BaseTrainer:
    def __init__(self, model, data, logger, config):
        self.model = model
        self.data = data
        self.logger = logger
        self.config = config

    def train_step(self):
        raise NotImplementedError('Train step method must be defined.')

    def train_epoch(self):
        pass


class BaseLogger:
    def __init__(self, root_dir):
        pass

    def log(self):
        pass
