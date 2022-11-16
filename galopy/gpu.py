import torch
from math import pi, factorial
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
    search.run()


if __name__ == "__main__":
    # search = NSxSearch('cuda', depth=3, n_ancilla_modes=2, n_ancilla_photons=1, n_success_measurements=1)
    # search.run()

    cz()

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
    # search.run()
