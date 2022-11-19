import torch
from math import pi, factorial, sqrt
from itertools import product
import numpy as np
import random
from galopy.genetic_algorithm import *
from galopy.nsx_search import *


def cz():
    matrix = np.array([[1., 0., 0., 0.],
                       [0., 1., 0., 0.],
                       [0., 0., 1., 0.],
                       [0., 0., 0., -1.]])
    # (3)----------
    # (2)----------
    # (1)----------
    # (0)----------
    basic_states = np.array([[0, 2],
                             [0, 3],
                             [1, 2],
                             [1, 3]])
    search = GeneticAlgorithm('cuda', matrix, input_basic_states=basic_states, depth=3,
                              n_ancilla_modes=2, n_ancilla_photons=0)
    search.run(1. / 9., 200, 800, 200, 0)


def cx():
    matrix = np.array([[1., 0., 0., 0.],
                       [0., 0., 0., 1.],
                       [0., 0., 1., 0.],
                       [0., 1., 0., 0.]])
    # (3)----------
    # (2)----------
    # (1)----------
    # (0)----------
    basic_states = np.array([[0, 2],
                             [0, 3],
                             [1, 2],
                             [1, 3]])
    search = GeneticAlgorithm('cuda', matrix, input_basic_states=basic_states, depth=5,
                              n_ancilla_modes=2, n_ancilla_photons=0)
    search.run(1. / 9., 200, 800, 200, 0)


def qft3():
    min_probability = 1.0

    n_parents = 800
    n_offsprings = 200
    n_generations = 500
    n_elite = 0

    matrix = np.array([[1. / sqrt(3.), 1. / sqrt(3.), 1. / sqrt(3.)],
                       [1. / sqrt(3.), -0.5 / sqrt(3.) + 0.5j, -0.5 / sqrt(3.) - 0.5j],
                       [1. / sqrt(3.), -0.5 / sqrt(3.) - 0.5j, -0.5 / sqrt(3.) + 0.5j]])
    # (2)----------
    # (1)----------
    # (0)----------
    basic_states = np.array([[2],
                             [1],
                             [0]])
    search = GeneticAlgorithm('cuda', matrix, input_basic_states=basic_states, depth=3,
                              n_ancilla_modes=0, n_ancilla_photons=0)
    search.run(min_probability, n_generations, n_parents, n_offsprings, n_elite)


def qft4():
    min_probability = 1.0

    n_parents = 1600
    n_offsprings = 400
    n_generations = 2000
    n_elite = 0

    matrix = np.array([[0.5, 0.5, 0.5, 0.5],
                       [0.5, 0.5j, -0.5, -0.5j],
                       [0.5, -0.5, 0.5, -0.5],
                       [0.5, -0.5j, -0.5, 0.5j]])
    # (3)----------
    # (2)----------
    # (1)----------
    # (0)----------
    basic_states = np.array([[3],
                             [2],
                             [1],
                             [0]])
    search = GeneticAlgorithm('cuda', matrix, input_basic_states=basic_states, depth=4,
                              n_ancilla_modes=0, n_ancilla_photons=0)
    search.run(min_probability, n_generations, n_parents, n_offsprings, n_elite)


if __name__ == "__main__":
    min_probability = 0.3

    n_parents = 1600
    n_offsprings = 400
    n_generations = 200
    n_elite = 0
    #
    # matrix = np.array([[1., 0., 0., 0.],
    #                    [0., 1. / sqrt(2.), 0., 1. / sqrt(2.)],
    #                    [0., 0., 1., 0.],
    #                    [0., 1. / sqrt(2.), 0., -1. / sqrt(2.)]])
    # # (3)----------
    # # (2)----------
    # # (1)----------
    # # (0)----------
    # basic_states = np.array([[0, 2],
    #                          [0, 3],
    #                          [1, 2],
    #                          [1, 3]])
    # search = GeneticAlgorithm('cuda', matrix, input_basic_states=basic_states, depth=5,
    #                           n_ancilla_modes=3, n_ancilla_photons=1)
    # search.run(min_probability, n_generations, n_parents, n_offsprings, n_elite)

    search = NSxSearch('cuda', depth=10, n_ancilla_modes=4, n_ancilla_photons=1, n_success_measurements=1)
    search.run(min_probability, n_generations, n_parents, n_offsprings, n_elite)

    # cz()
    # qft3()

    # matrix = np.array([[1., 0., 0., 0.],
    #                    [0., 1., 0., 0.],
    #                    [0., 0., 1., 0.],
    #                    [0., 0., 0., -1.],
    #                    [0., 0., 0., 0.],
    #                    [0., 0., 0., 0.],
    #                    [0., 0., 0., 0.],
    #                    [0., 0., 0., 0.],
    #                    [0., 0., 0., 0.],
    #                    [0., 0., 0., 0.]])
    # # (3)----------
    # # (2)----------
    # # (1)----------
    # # (0)----------
    # input_basic_states = np.array([[0, 2],
    #                                [0, 3],
    #                                [1, 2],
    #                                [1, 3]])
    # output_basic_states = np.array([[0, 2],
    #                                 [0, 3],
    #                                 [1, 2],
    #                                 [1, 3],
    #                                 [0, 0],
    #                                 [1, 1],
    #                                 [2, 2],
    #                                 [3, 3],
    #                                 [0, 1],
    #                                 [2, 3]])
    # search = GeneticAlgorithm('cuda', matrix, input_basic_states=input_basic_states,
    #                           output_basic_states=output_basic_states, depth=8,
    #                           n_ancilla_modes=4, n_ancilla_photons=2)
    # search.run(min_probability, n_generations, n_parents, n_offsprings, n_elite)
