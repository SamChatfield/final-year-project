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


def eaMuPlusLambda(
    population,
    toolbox,
    mu,
    lambda_,
    cxpb,
    mutpb,
    ngen,
    stats=None,
    halloffame=None,
    verbose=__debug__,
):
    """A modified version of deap.algorithms.eaMuPlusLambda.

    The difference is that each generation _all_ individuals have their fitness
    re-evaluated rather than only the new ones. This is because the fitness of
    of the same individual will change from generation to generation because the
    weights of the discriminator network will have changed.
    """
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = population
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = population + offspring
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


class EA:
    # pylint: disable=no-member

    def __init__(self, discriminator, pop_size=50, states=3, symbols=3, pool_size=None):
        # Maximise fitness (amount it will fool discriminator)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        # Individual is 2-tuple of (transition, emission) ndarrays
        creator.create("Individual", tuple, fitness=creator.FitnessMin)

        self._discriminator = discriminator
        self._pop_size = pop_size
        self._states = states
        self._symbols = symbols

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

    def run(self, gens, offpr, cxpb, mutpb, use_hof=False):
        pop = self.toolbox.population(n=self._pop_size)

        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("mean", np.mean)
        stats.register("min", np.min)

        hof = tools.HallOfFame(maxsize=10, similar=np.array_equal) if use_hof else None

        final_pop, logbook = eaMuPlusLambda(
            pop,
            self.toolbox,
            mu=self._pop_size,
            lambda_=math.floor(offpr * self._pop_size),
            cxpb=cxpb,
            mutpb=mutpb,
            ngen=gens,
            stats=stats,
            halloffame=hof,
            verbose=True,
        )

        return final_pop, hof, logbook

    def cleanup(self):
        if self._pool:
            self._pool.close()


if __name__ == "__main__":
    ea = EA(None)
    ea.run()
