import numpy as np
from galopy.gd import CircuitSearch
import galopy.gd.topology as tl
from math import sqrt

if __name__ == '__main__':
    # Gate represented as a matrix
    target_matrix = np.array([[1. / sqrt(2.)],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [1. / sqrt(2.)],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.]])
    # target_matrix = np.array([[1. / sqrt(3.)],
    #                           [0],
    #                           [1. / sqrt(3.)],
    #                           [1. / sqrt(3.)],
    #                           [0.],
    #                           [0.],
    #                           [0.],
    #                           [0.],
    #                           [0.],
    #                           [0.]])

    # State modes:
    # (5)----------
    # (4)----------
    # (3)----------
    # (2)----------
    # (1)----------
    # (0)----------
    input_basic_states = np.array([[1, 3, 5]])
    output_basic_states = np.array([[0, 2, 4],
                                    [0, 2, 5],
                                    [0, 3, 4],
                                    [0, 3, 5],
                                    [1, 2, 4],
                                    [1, 2, 5],
                                    [1, 3, 4],
                                    [1, 3, 5],
                                    [0, 0, 0],
                                    [0, 0, 1],
                                    [0, 0, 2],
                                    [0, 0, 3],
                                    [0, 0, 4],
                                    [0, 0, 5],
                                    [0, 1, 1],
                                    [0, 1, 2],
                                    [0, 1, 3],
                                    [0, 1, 4],
                                    [0, 1, 5],
                                    [0, 2, 2],
                                    [0, 2, 3],
                                    # [0, 2, 4],
                                    # [0, 2, 5],
                                    [0, 3, 3],
                                    # [0, 3, 4],
                                    # [0, 3, 5],
                                    [0, 4, 4],
                                    [0, 4, 5],
                                    [0, 5, 5],
                                    [1, 1, 1],
                                    [1, 1, 2],
                                    [1, 1, 3],
                                    [1, 1, 4],
                                    [1, 1, 5],
                                    [1, 2, 2],
                                    [1, 2, 3],
                                    # [1, 2, 4],
                                    # [1, 2, 5],
                                    [1, 3, 3],
                                    # [1, 3, 4],
                                    # [1, 3, 5],
                                    [1, 4, 4],
                                    [1, 4, 5],
                                    [1, 5, 5],
                                    [2, 2, 2],
                                    [2, 2, 3],
                                    [2, 2, 4],
                                    [2, 2, 5],
                                    [2, 3, 3],
                                    [2, 3, 4],
                                    [2, 3, 5],
                                    [2, 4, 4],
                                    [2, 4, 5],
                                    [2, 5, 5],
                                    [3, 3, 3],
                                    [3, 3, 4],
                                    [3, 3, 5],
                                    [3, 4, 4],
                                    [3, 4, 5],
                                    [3, 5, 5],
                                    [4, 4, 4],
                                    [4, 4, 5],
                                    [4, 5, 5],
                                    [5, 5, 5]])

    # Ancilla modes:
    # (3)----------
    # (2)----------
    # (1)----------
    # (0)----------
    ancilla_state = np.array([0, 1, 2])
    measurements = np.array([[0, 1, 2],
                             [0, 1, 3]])

    # Create an instance of search
    search = CircuitSearch(target_matrix, input_basic_states=input_basic_states,
                           output_basic_states=output_basic_states, n_ancilla_modes=4, measurements=measurements,
                           ancilla_state=ancilla_state, topology=tl.Parallel,
                           device='cpu')
    # search = CircuitSearch(target_matrix, input_basic_states, n_ancilla_modes=3, measurements=measurements,
    #                        ancilla_state=ancilla_state, output_basic_states=output_basic_states, topology=tl.Parallel,
    #                        device='cpu')

    # Launch the search!
    circuit = search.run(min_probability=1 / 54, n_epochs=3000, print_info=True)
    # circuit = search.run(min_probability=2 / 27, n_epochs=2000, print_info=True)

    # Print result
    print("Circuit:")
    circuit.print()

    # Save result
    circuit.to_loqc_tech("result.json")
