import multiprocessing

from functools import partial

import numpy as np
from deap import base, creator, tools

# POOL_SIZE = multiprocessing.cpu_count() // 2
POOL_SIZE = 4
print(f'Pool Size = {POOL_SIZE}')
STATES = 5
SYMBOLS = 3


def setup_creator():
    # Maximise fitness (amount it will fool discriminator)
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    # Individual is 2-tuple of (transition, emission) ndarrays
    creator.create('Individual', tuple, fitness=creator.FitnessMax)


def evaluate(ind):
    # Do CNN model training batch and take 1 - acc as fitness
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


def main():
    setup_creator()

    # Init toolbox
    toolbox = base.Toolbox()

    # Set up multiprocessing on toolbox
    pool = multiprocessing.Pool(processes=POOL_SIZE)
    toolbox.register('map', pool.map)

    # Complete toolbox setup
    setup_toolbox(toolbox)

    test_pop = toolbox.population(10)
    print(f'Test Pop:\n{test_pop}\n')

    pool.close()


if __name__ == '__main__':
    main()
