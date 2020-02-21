import math
import multiprocessing
import random
from functools import partial

import numpy as np
from deap import algorithms, base, creator, tools

import hmm


def ind_str(ind):
    t, e = ind
    return f"{t}\n{e}\nFitness: {ind.fitness}"


def evaluate(ind, discriminator):
    real_hmm = discriminator._real_hmm
    data_gen = discriminator._train_data_generator

    # Build HMM from individual
    t_mat, e_mat = ind
    ind_hmm = hmm.HMM(real_hmm.x, real_hmm.y, t_mat, e_mat, real_hmm.s)

    data_gen = discriminator._train_data_generator
    X, y = data_gen.create_batch(ind_hmm)

    # Train discriminator on batch
    metrics = discriminator._model.train_on_batch(X, y)
    # Metrics is numpy array with: [loss, accuracy]

    # Return fitness proportional to training accuracy for batch
    return (metrics[1],)


def mutate(ind, indpb, states, symbols):
    # Mutate each row in the transition and emission matrices with probability indpb

    # Transition matrix
    for i in range(states):
        if random.random() < indpb:
            # Mutate
            mut = np.random.dirichlet(np.ones(states))
            ind[0][i] += mut
            ind[0][i] /= ind[0][i].sum()

    # Emission matrix
    for i in range(states):
        if random.random() < indpb:
            # Mutate
            mut = np.random.dirichlet(np.ones(symbols))
            ind[1][i] += mut
            ind[1][i] /= ind[1][i].sum()

    return (ind,)


def crossover(ind1, ind2):
    # Crossover two individuals
    # TODO: Implement
    pass


class EA:
    # pylint: disable=no-member

    def __init__(self, discriminator, pop_size=50, states=3, symbols=3, pool_size=None):
        # Maximise fitness (amount it will fool discriminator)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        # Individual is 2-tuple of (transition, emission) ndarrays
        creator.create("Individual", tuple, fitness=creator.FitnessMax)

        self._discriminator = discriminator
        self._pop_size = pop_size
        self._states = states
        self._symbols = symbols
        print(f"Discriminator metrics: {discriminator._model.metrics_names}")
        # Set up multiprocessing pool
        self._pool = multiprocessing.Pool(processes=pool_size) if pool_size else None

        # Init toolbox
        self.toolbox = base.Toolbox()
        # Complete toolbox setup
        self._setup_toolbox()

    def _setup_toolbox(self):
        # Register map to use multiprocessing pool
        if self._pool:
            self.toolbox.register("map", self._pool.map)

        # Transition matrix generation
        self.toolbox.register(
            "trans_mat", np.random.dirichlet, np.ones(self._states), self._states
        )
        # Emission matrix generation
        self.toolbox.register(
            "emiss_mat", np.random.dirichlet, np.ones(self._symbols), self._states
        )
        # Generate Individual from transition and matrix generator in cycle
        self.toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            (self.toolbox.trans_mat, self.toolbox.emiss_mat),
        )

        # Population generation
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        # Crossover
        self.toolbox.register("mate", crossover)
        # Mutation
        mut_rate = 1 / (2 * self._states)
        self.toolbox.register(
            "mutate", mutate, indpb=mut_rate, states=self._states, symbols=self._symbols
        )
        # Selection
        self.toolbox.register("select", tools.selTournament, tournsize=2)
        # Fitness evaluation
        self.toolbox.register("evaluate", evaluate, discriminator=self._discriminator)

    def run(self):
        pop = self.toolbox.population(n=self._pop_size)

        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)

        final_pop, _ = algorithms.eaMuPlusLambda(
            pop,
            self.toolbox,
            mu=self._pop_size,
            lambda_=math.floor(0.5 * self._pop_size),
            cxpb=0.0,
            mutpb=1.0,
            ngen=50,
            stats=stats,
            verbose=True,
        )

        return final_pop

    def cleanup(self):
        if self._pool:
            self._pool.close()


if __name__ == "__main__":
    ea = EA(None)
    ea.run()
