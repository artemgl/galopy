import numpy as np
from galopy.gd import CircuitSearch
import galopy.gd.topology as tl

if __name__ == '__main__':
    # Gate represented as a matrix
    target_matrix = np.array([[1., 0., 0., 0.],
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

    # Ancilla modes:
    # (1)----------
    # (0)----------
    ancilla_state = np.array([0, 1])
    measurements = np.array([[0, 0],
                             [0, 1],
                             [1, 1]])

    # Create an instance of search
    search = CircuitSearch(target_matrix, input_basic_states, n_ancilla_modes=2, measurements=measurements,
                           ancilla_state=ancilla_state, output_basic_states=output_basic_states, topology=tl.Parallel,
                           device='cuda')

    # Launch the search!
    circuit = search.run(min_probability=2 / 27, n_epochs=2000, print_info=True)

    # Print result
    print("Circuit:")
    circuit.print()

    # Save result
    # circuit.to_loqc_tech("result.json")
