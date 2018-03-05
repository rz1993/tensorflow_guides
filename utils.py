import matplotlib.pyplot as plt


def plot_curves(**curves):
    '''
    Utility function for plotting multiple curves:
    Args:
        - curves : a dictionary of dictionaries containing
                   curve_label: curves data and plotting parameters
    '''
    for key in curves:
        params = curves[key]
        plt.plot(params.get('x', range(len(params['y']))),
                 params['y'],
                 color=params['color'],
                 label=key)
    plt.legend()
    plt.show()


class Batcher:
    def __init__(self, data, batch_size):
        self._data = data
        self._batch_size = batch_size

        # Make sure we can batch size is compatible
        assert data.shape[0] % batch_size == 0

        self._n_batches = data.shape[0] // batch_size
        self._idxs = np.arange(data.shape[0])

    def add_noise(self, batch):
        return batch + np.random.normal(size=batch.shape)

    def get_batches(self, add_noise=False):
        batch_size = self._batch_size

        for i in range(self._n_batches):
            beg = i * batch_size
            end = beg + batch_size
            batch_idxs = self._idxs[beg:end]
            batch = self._data[batch_idxs]
            if add_noise:
                batch = self.add_noise(batch)
            yield batch

        np.random.shuffle(self._idxs)
