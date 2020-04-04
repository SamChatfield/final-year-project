from string import ascii_lowercase

import hmm


def run(discriminator, states, symbols, gens):
    best_hmm = None
    best_acc = 1.0

    for i in range(gens):
        # Generate a random HMM of the correct shape
        x = states
        y = ascii_lowercase[:symbols]
        s = [1.0] + [0.0] * (x - 1)
        rand_hmm = hmm.random_hmm(x, y, s)

        # Evaluate with discriminator
        X, y = discriminator._train_data_generator.create_batch(rand_hmm)
        metrics = discriminator._model.train_on_batch(X, y)
        acc = metrics[1]

        # Minimise accuracy
        if acc < best_acc:
            best_hmm = rand_hmm
            best_acc = acc

    return best_hmm, best_acc
