import math
import multiprocessing
import random
from functools import partial

import numpy as np
from deap import algorithms, base, creator, tools

# POOL_SIZE = multiprocessing.cpu_count() // 2
POOL_SIZE = 4
print(f'Pool Size = {POOL_SIZE}')
STATES = 5
SYMBOLS = 3
POP_SIZE = 10


def ind_str(ind):
    t, e = ind
    return f'{t}\n{e}\n{ind.fitness}'


def evaluate(ind):
    # Do CNN model training batch and take 1 - acc as fitness
    # TODO: Remove placeholder
    return random.random(),


def mutate(ind, indpb):
    # Mutate each row in the transition and emission matrices with probability indpb

    # Transition matrix
    for i in range(STATES):
        if random.random() < indpb:
            # Mutate
            mut = np.random.dirichlet(np.ones(STATES))
            ind[0][i] += mut
            ind[0][i] /= ind[0][i].sum()

    # Emission matrix
    for i in range(STATES):
        if random.random() < indpb:
            # Mutate
            mut = np.random.dirichlet(np.ones(SYMBOLS))
            ind[1][i] += mut
            ind[1][i] /= ind[1][i].sum()

    return ind,


def crossover(ind1, ind2):
    # Crossover two individuals
    # TODO: Implement
    pass


def setup_toolbox(toolbox):
    # Transition matrix generation
    toolbox.register(
        'trans_mat',
        np.random.dirichlet,
        np.ones(STATES),
        STATES
    )
    # Emission matrix generation
    toolbox.register(
        'emiss_mat',
        np.random.dirichlet,
        np.ones(SYMBOLS),
        STATES
    )
    # Generate Individual from transition and matrix generator in cycle
    toolbox.register(
        'individual',
        tools.initCycle,
        creator.Individual,
        (toolbox.trans_mat, toolbox.emiss_mat)
    )

    # Population generation
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # Crossover
    toolbox.register('mate', crossover)
    # Mutation
    toolbox.register('mutate', mutate, indpb=1/(2*STATES))
    # Selection
    toolbox.register('select', tools.selTournament, tournsize=2)
    # Fitness evaluation
    toolbox.register('evaluate', evaluate)


def main():
    # Maximise fitness (amount it will fool discriminator)
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    # Individual is 2-tuple of (transition, emission) ndarrays
    creator.create('Individual', tuple, fitness=creator.FitnessMax)

    # Init toolbox
    toolbox = base.Toolbox()

    # Set up multiprocessing pool
    pool = multiprocessing.Pool(processes=POOL_SIZE)
    toolbox.register('map', pool.map)

    # Complete toolbox setup
    setup_toolbox(toolbox)

    # Create initial population
    pop = toolbox.population(n=POP_SIZE)

    ind1 = pop[0]
    print(f'Ind 1:\n{ind1[0]}\n{ind1[1]}\n')

    mut1, = toolbox.mutate(ind1)
    print(f'Mut 1:\n{mut1[0]}\n{mut1[1]}\n')

    final_pop, _ = algorithms.eaMuPlusLambda(
        pop,
        toolbox,
        mu=POP_SIZE,
        lambda_=math.floor(0.5*POP_SIZE),
        cxpb=0.0,
        mutpb=1.0,
        ngen=50,
        verbose=True
    )

    for i, ind in enumerate(final_pop):
        print(f'Ind {i}:\n{ind_str(ind)}\n')

    pool.close()


if __name__ == '__main__':
    main()
