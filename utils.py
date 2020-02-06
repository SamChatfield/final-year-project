import math

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
        self._symbols = ''.join(self._real_hmm.y)

        self._init_tokenizer()

    def _init_tokenizer(self):
        self._tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
        self._tokenizer.fit_on_texts(self._symbols)

    def __len__(self):
        # Returns num. batches per epoch
        return self._epoch_size

    def __getitem__(self, index):
        # Returns a whole batch
        # 50% of batch real (from real_hmm), 50% fake (from rand_hmm)???
        # New random HMM generated for each each???
        # Batch size then equal to number of samples from each HMM
        num_rand_samples = math.ceil(0.5 * self._batch_size)
        num_real_samples = math.floor(0.5 * self._batch_size)

        # 1. Generate rand_hmm
        rand_hmm = hmm.random_hmm()

        # 2. Sample 0.5 x batch_size sequences from rand_hmm
        rand_samples = [
            rand_hmm.simulate(self._seq_len, reset_before=True)[1]
            for _ in range(num_rand_samples)
        ]
        rand_samples = [''.join(s) for s in rand_samples]
        rand_labels = np.zeros(num_rand_samples)

        # 3. Sample 0.5 x batch_size sequences from real_hmm
        real_samples = [
            self._real_hmm.simulate(self._seq_len, reset_before=True)[1]
            for _ in range(num_real_samples)
        ]
        real_samples = [''.join(s) for s in real_samples]
        real_labels = np.ones(num_real_samples)

        # 4. Tokenise both sequences
        rand_samples_enc = self._encode_hmm_outputs(rand_samples)
        real_samples_enc = self._encode_hmm_outputs(real_samples)

        try:
            X = np.concatenate((rand_samples_enc, real_samples_enc))
            y = np.concatenate((rand_labels, real_labels))
        except:
            print(rand_samples_enc)
            print(real_samples_enc)
            print(rand_labels)
            print(real_labels)
            raise

        # Shuffle the samples
        p = np.random.permutation(self._batch_size)
        X = X[p]
        y = y[p]

        return X, y

    def _encode_hmm_outputs(self, hmm_outputs):
        tokens = self._tokenizer.texts_to_sequences(hmm_outputs)
        tokens = np.array(tokens) - 1
        onehot = keras.utils.to_categorical(
            tokens,
            num_classes=len(self._symbols)
        )
        return onehot

    def input_shape(self):
        return (self._seq_len, len(self._symbols))


def plot_model(model, to_file):
    return keras.utils.plot_model(model, to_file=to_file, show_shapes=True)


def plot_acc(history, to_file=None, val=False):
    plt.plot(history.history['accuracy'])
    if val: plt.plot(history.history['val_accuracy'])

    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    if val:
        plt.legend(['Train', 'Val'], loc='center right')
    else:
        plt.legend(['Train'], loc='center right')
    
    if to_file:
        plt.savefig(to_file)
    
    plt.show()


def plot_loss(history, to_file=None, val=False):
    plt.plot(history.history['loss'])
    if val: plt.plot(history.history['val_loss'])

    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    if val:
        plt.legend(['Train', 'Val'], loc='center right')
    else:
        plt.legend(['Train'], loc='center right')

    if to_file:
        plt.savefig(to_file)

    plt.show()
