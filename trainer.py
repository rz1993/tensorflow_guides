from base import BaseTrainer


class SimpleTrainer(BaseTrainer):
    def __init__(self, config):
        super(SimpleTrainer, self).__init__(config)

    def train(self, model, data):
        losses = []

        for _ in range(self.config['num_epochs']):
            pass

    def train_step(self, model, data):
        batch_inputs = model.inputs
        batch_data = data.next_batch(self.config['batch_size'])

        if isinstance(batch_data, list):
            assert len(batch_data) == len(batch_inputs)
        else:
            raise Exception('Batch data and inputs should be lists of '
                            'arrays and input tensors respectively.')

        loss = model.loss
        train_op = model.train_op

        loss_, _ = self.sess.run([loss, train_op],
            feed_dict=dict(zip(batch_inputs, batch_data)))

        model.incr_step.eval(self.sess)

        return loss_

    def train_epoch(self, model, data):
        pass
