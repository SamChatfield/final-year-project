import numpy as np

import hmm
import utils
from cnn import CNNModel3 as CNNModel
from discriminator import Discriminator
from ea import EA

POOL_SIZE = 4


def init_real_hmm():
    # Set the fixed parameters of the "real" HMM
    x = 5
    y = "abc"
    s = [1.0, 0.0, 0.0, 0.0, 0.0]

    # Create "real" HMM with random transition and emission matrices
    # real_hmm = hmm.random_hmm(x, y, s)
    real_hmm = hmm.random_hmm(x, y, s)

    return real_hmm


def init_discriminator(real_hmm):
    # Set training parameters
    epochs = 20
    epoch_size = 100
    batch_size = 100
    seq_len = 20

    # Create real HMM data generator
    # train_data_gen = utils.HMMDataGenerator(real_hmm, epoch_size, batch_size, seq_len)
    # model = CNNModel(train_data_gen.input_shape())

    discriminator = Discriminator(real_hmm, epoch_size, batch_size, seq_len)

    # model.fit_generator(generator=train_data_gen, epochs=epochs)
    discriminator.initial_train(epochs)

    return discriminator


def main():
    # Initialise "real" HMM randomly
    real_hmm = init_real_hmm()

    # Initialise and train the neural network discriminator
    discriminator = init_discriminator(real_hmm)

    # Initialise EA toolbox
    # ea = EA(discriminator, pool_size=POOL_SIZE)
    ea = EA(discriminator)
    # Run EA
    final_pop = ea.run()
    # Clean up EA
    ea.cleanup()


if __name__ == "__main__":
    main()
