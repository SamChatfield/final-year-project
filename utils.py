import math
import multiprocessing

import keras
import matplotlib.pyplot as plt
import numpy as np

import hmm


class HMMDataGenerator(keras.utils.Sequence):
    def __init__(self, real_hmm, epoch_size, batch_size, seq_len):
        super().__init__()
        self._real_hmm = real_hmm
        self._epoch_size = epoch_size
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._symbols = "".join(self._real_hmm.y)

        self._init_tokenizer()

    def _init_tokenizer(self):
        self._tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
        self._tokenizer.fit_on_texts(self._symbols)

    def __len__(self):
        # Returns num. batches per epoch
        return self._epoch_size

    def __getitem__(self, index):
        # Generate rand_hmm
        rand_hmm = hmm.random_hmm(
            x=self._real_hmm.x, y=self._symbols, s=self._real_hmm.s
        )
        return self.create_batch(rand_hmm)

    def create_batch(self, other_hmm):
        # Returns a whole batch
        # 50% of batch real (from real_hmm), 50% fake (from rand_hmm)???
        # New random HMM generated for each each???
        # Batch size then equal to number of samples from each HMM
        num_other_samples = math.ceil(0.5 * self._batch_size)
        num_real_samples = math.floor(0.5 * self._batch_size)

        # Sample 0.5 x batch_size sequences from other_hmm
        other_samples = [
            other_hmm.simulate(self._seq_len, reset_before=True)[1]
            for _ in range(num_other_samples)
        ]
        other_samples = ["".join(s) for s in other_samples]
        other_labels = np.zeros(num_other_samples)

        # Sample 0.5 x batch_size sequences from real_hmm
        real_samples = [
            self._real_hmm.simulate(self._seq_len, reset_before=True)[1]
            for _ in range(num_real_samples)
        ]
        real_samples = ["".join(s) for s in real_samples]
        real_labels = np.ones(num_real_samples)

        # One-hot encode both sequences
        other_samples_enc = self._encode_hmm_outputs(other_samples)
        real_samples_enc = self._encode_hmm_outputs(real_samples)

        # Concatenate the sequences and labels from both HMMs
        X = np.concatenate((other_samples_enc, real_samples_enc))
        y = np.concatenate((other_labels, real_labels))

        # Shuffle the samples
        p = np.random.permutation(self._batch_size)
        X = X[p]
        y = y[p]

        return X, y

    def _encode_hmm_outputs(self, hmm_outputs):
        tokens = self._tokenizer.texts_to_sequences(hmm_outputs)
        tokens = np.array(tokens) - 1
        onehot = keras.utils.to_categorical(tokens, num_classes=len(self._symbols))
        return onehot

    def input_shape(self):
        return (self._seq_len, len(self._symbols))


# Sub-class multiprocessing.pool.Pool to allow creation of non-daemon processes
# This is necessary to be able to create processes that can spawn more processes
# This is used in experiments to run multiple instances of the algorithm at once while
# still being able to use multiprocessing to train the NN discriminator


class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


class MyPool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs["context"] = NoDaemonContext()
        super(MyPool, self).__init__(*args, **kwargs)


def plot_model(model, to_file):
    return keras.utils.plot_model(model, to_file=to_file, show_shapes=True)


def plot_acc(history, to_file=None, val=False):
    plt.plot(history.history["accuracy"])
    if val:
        plt.plot(history.history["val_accuracy"])

    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")

    if val:
        plt.legend(["Train", "Val"], loc="center right")
    else:
        plt.legend(["Train"], loc="center right")

    if to_file:
        plt.savefig(to_file)

    plt.show()


def plot_loss(history, to_file=None, val=False):
    plt.plot(history.history["loss"])
    if val:
        plt.plot(history.history["val_loss"])

    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")

    if val:
        plt.legend(["Train", "Val"], loc="center right")
    else:
        plt.legend(["Train"], loc="center right")

    if to_file:
        plt.savefig(to_file)

    plt.show()


def callbacks(model_name):
    model_checkpoint_cb = keras.callbacks.ModelCheckpoint(
        f"models/weights-{model_name}.h5",
        monitor="loss",
        save_best_only=True,
        save_weights_only=True,
    )

    return [model_checkpoint_cb]
