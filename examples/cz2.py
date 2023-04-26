import numpy as np
from galopy import *

if __name__ == "__main__":
    # Initialize parameters
    min_probability = 2 / 27
    n_offsprings = 400
    n_elite = 200
    n_generations = 1000

    # Gate represented as a matrix
    matrix = np.array([[1., 0., 0., 0.],
                       [0., 1., 0., 0.],
                       [0., 0., 1., 0.],
                       [0., 0., 0., -1.],
                       [0., 0., 0., 0.],
                       [0., 0., 0., 0.],
                       [0., 0., 0., 0.],
                       [0., 0., 0., 0.],
                       [0., 0., 0., 0.],
                       [0., 0., 0., 0.]])
    # State modes:
    # (3)----------
    # (2)----------
    # (1)----------
    # (0)----------
    input_basic_states = np.array([[0, 2],
                                   [0, 3],
                                   [1, 2],
                                   [1, 3]])
    output_basic_states = np.array([[0, 2],
                                    [0, 3],
                                    [1, 2],
                                    [1, 3],
                                    [0, 0],
                                    [1, 1],
                                    [2, 2],
                                    [3, 3],
                                    [0, 1],
                                    [2, 3]])
    # Create an instance of search
    search = CircuitSearch('cpu', matrix, input_basic_states=input_basic_states,
                           output_basic_states=output_basic_states, depth=8,
                           n_ancilla_modes=4, n_ancilla_photons=2)
    # Launch the search!
    circuit = search.run(min_probability, n_generations, n_offsprings, n_elite)

    # Save result
    circuit.to_loqc_tech("result.json")
