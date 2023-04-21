import numpy as np
from galopy.circuit_search import *

if __name__ == "__main__":
    # Initialize parameters
    min_probability = 1. / 9.
    n_offsprings = 1200
    n_elite = 800
    n_generations = 300

    # Gate represented as a matrix
    matrix = np.array([[1., 0., 0., 0.],
                       [0., 1., 0., 0.],
                       [0., 0., 1., 0.],
                       [0., 0., 0., -1.]])
    # State modes:
    # (3)----------
    # (2)----------
    # (1)----------
    # (0)----------
    basic_states = np.array([[0, 2],
                             [0, 3],
                             [1, 2],
                             [1, 3]])
    # Create an instance of search
    search = CircuitSearch('cpu', matrix, input_basic_states=basic_states, depth=3,
                           n_ancilla_modes=2, n_ancilla_photons=0)
    # Launch the search!
    search.run(min_probability, n_generations, n_offsprings, n_elite)
