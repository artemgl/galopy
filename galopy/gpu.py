import torch
from math import pi, factorial
from itertools import product
import numpy as np
import random
from galopy.genetic_algorithm import *


if __name__ == "__main__":
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
    #
    # a = torch.tensor([0, 1, 2], device='cuda')
    # indices = sum(((a,), (0,)), ())
    #
    # x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device='cuda')
    # print(x)
    # print(x[indices])
