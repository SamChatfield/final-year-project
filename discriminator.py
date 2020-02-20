import multiprocessing

import utils
from cnn import CNNModel3 as CNNModel


class Discriminator:
    def __init__(
        self, real_hmm, epoch_size, batch_size, sequence_length, pool_size=None
    ):
        self._real_hmm = real_hmm
        self._epoch_size = epoch_size
        self._batch_size = batch_size
        self._sequence_length = sequence_length
        self._pool_size = pool_size

        self._train_data_generator = utils.HMMDataGenerator(
            self._real_hmm, self._epoch_size, self._batch_size, self._sequence_length
        )

        self._model = CNNModel(self._train_data_generator.input_shape())

    def initial_train(self, epochs):
        print("\nPre-training discriminator:")
        if self._pool_size:
            self._model.fit_generator(
                generator=self._train_data_generator,
                epochs=epochs,
                use_multiprocessing=True,
                workers=self._pool_size,
            )
        else:
            self._model.fit_generator(
                generator=self._train_data_generator, epochs=epochs
            )
        print()
