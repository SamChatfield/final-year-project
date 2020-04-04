import json
import string
from datetime import datetime

import deap
import numpy as np

import hmm
from discriminator import Discriminator
from ea import EA
import random_search

DEFAULT_PARAMS = {
    # Discriminator CNN model
    "model": "CNNModel3",
    # Algorithm Parameters
    "states": 5,
    "symbols": 5,
    "epochs": 10,
    "epoch_size": 500,
    "batch_size": 200,
    "seq_len": 20,
    "pop_size": 25,
    "gens": 50,
    "offspring_prop": 1.0,
    "cx_prob": 0.0,
    "mut_fn": "uniform",
    "mut_prob": 1.0,
    "mut_rate": None,  # None - default to 1/N where N is number of genes
    # Implementation Parameters
    "_pool_size": 4,
    "_random_search": True,  # Also run an elitist random search over #gens to compare performance
}


def param_assert(params):
    assert params["states"] > 0
    assert 0 < params["symbols"] <= 26
    assert 0.0 <= params["offspring_prop"] <= 1.0
    assert 0.0 <= params["cx_prob"] <= 1.0
    assert 0.0 <= params["mut_prob"] <= 1.0
    assert (params["mut_rate"] is None) or (0.0 <= params["mut_rate"] <= 1.0)


def run(param_subset):
    # Overwrite the default values of the provided parameters
    params = {**DEFAULT_PARAMS, **param_subset}

    print(params)
    param_assert(params)

    x = params["states"]
    y = string.ascii_lowercase[: params["symbols"]]
    s = [1.0] + [0.0] * (x - 1)
    # Random HMM that will act as the 'true' underlying distribution
    real_hmm = hmm.random_hmm(x, y, s)
    # Different random HMM that will be used to benchmark the best solution we find
    rand_hmm = hmm.random_hmm(x, y, s)

    d = Discriminator(
        real_hmm,
        params["epoch_size"],
        params["batch_size"],
        params["seq_len"],
        model=params["model"],
        pool_size=params["_pool_size"],
    )

    print("Pre-training discriminator...")
    accs = d.initial_train(params["epochs"])
    acc = accs[-1]
    print(f"Pre-trained discriminiator accuracy: {acc}")

    g = EA(
        discriminator=d,
        pop_size=params["pop_size"],
        states=x,
        symbols=len(y),
        offpr=params["offspring_prop"],
        cxpb=params["cx_prob"],
        mut_fn=params["mut_fn"],
        mutpb=params["mut_prob"],
        mut_rate=params["mut_rate"],
    )

    print("Running generator...")
    final_pop, _, logbook = g.run(params["gens"])

    best_ind = deap.tools.selBest(final_pop, 1)[0]
    best_hmm = hmm.HMM(x, np.array(list(y)), best_ind[0], best_ind[1], np.array(s))

    if params["_random_search"]:
        print("Running random search benchmark...")
        rs_best_hmm, rs_best_acc = random_search.run(
            d, params["states"], params["symbols"], params["gens"]
        )
    else:
        rs_best_hmm, rs_best_acc = None, None

    return real_hmm, best_hmm, rand_hmm, rs_best_hmm, logbook


def experiment(params, runs):
    all_params = {**DEFAULT_PARAMS, **params}
    do_rand_search = all_params["_random_search"]

    mean_fitnesses = []
    best_l2s = []
    rand_l2s = []
    if do_rand_search:
        rs_l2s = []

    for i in range(runs):
        print(f"Run {i+1}")
        real_hmm, best_hmm, rand_hmm, rs_best_hmm, logbook = run(params)

        best_l2 = hmm.total_l2_diff(real_hmm, best_hmm)
        rand_l2 = hmm.total_l2_diff(real_hmm, rand_hmm)
        if do_rand_search:
            rs_l2 = hmm.total_l2_diff(real_hmm, rs_best_hmm)

        mean_fitnesses.append(logbook.select("mean"))
        best_l2s.append(best_l2)
        rand_l2s.append(rand_l2)
        extra_msg = ""
        if do_rand_search:
            rs_l2s.append(rs_l2)
            extra_msg = f", RandSearch L2: {rs_l2}"

        print(f"Best L2: {best_l2}, Rand L2: {rand_l2}{extra_msg}")

    exp_data = {
        "params": all_params,
        "mean_fitnesses": mean_fitnesses,
        "best_l2s": best_l2s,
        "rand_l2s": rand_l2s,
    }
    if do_rand_search:
        exp_data["rs_l2s"] = rs_l2s

    exp_file = f'experiments/exp_{datetime.now().strftime("%y%m%d-%H%M%S%f")}.json'
    with open(exp_file, "w") as f:
        json.dump(exp_data, f, indent=4)

    return exp_data


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
