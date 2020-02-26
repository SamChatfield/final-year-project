import string

import deap
import numpy as np

import hmm
from discriminator import Discriminator
from ea import EA

DEFAULT_PARAMS = {
    "states": 3,
    "symbols": 5,
    "epochs": 10,
    "epoch_size": 500,
    "batch_size": 200,
    "seq_len": 20,
    "pool_size": 8,
    "pop_size": 100,
    "gens": 50,
    "offspring_prop": 0.25,
    "cx_prob": 0.0,
    "mut_prob": 1.0,
}


def param_assert(params):
    assert params["states"] > 0
    assert 0 < params["symbols"] <= 26


def run(params):
    print(params)
    param_assert(params)

    x = params["states"]
    y = string.ascii_lowercase[: params["symbols"]]
    s = [1.0] + [0.0] * (x - 1)
    real_hmm = hmm.random_hmm(x, y, s)

    d = Discriminator(
        real_hmm,
        params["epoch_size"],
        params["batch_size"],
        params["seq_len"],
        pool_size=params["pool_size"],
    )

    d.initial_train(params["epochs"])

    g = EA(d, params["pop_size"], states=x, symbols=len(y))

    final_pop, hall_of_fame = g.run(
        params["gens"], params["offspring_prop"], params["cx_prob"], params["mut_prob"],
    )

    best_ind = deap.tools.selBest(final_pop, 1)[0]
    best_hmm = hmm.HMM(x, np.array(list(y)), best_ind[0], best_ind[1], np.array(s))

    rand_hmm = hmm.random_hmm(x, y, s)

    best_l2 = hmm.total_l2_diff(real_hmm, best_hmm)
    rand_l2 = hmm.total_l2_diff(real_hmm, rand_hmm)

    return real_hmm, best_hmm, best_l2, rand_l2


def main():
    real_hmm, best_hmm, best_l2 = run(DEFAULT_PARAMS)
    print(
        f"""
        Real HMM: {real_hmm}

        Best HMM: {best_hmm}

        Best L2:  {best_l2}
        """
    )


if __name__ == "__main__":
    main()
