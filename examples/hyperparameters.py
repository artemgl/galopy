import numpy as np
from galopy.gd import MultiCircuitSearch
import galopy.gd.topology as tl
from itertools import product


if __name__ == '__main__':
    # Gate represented as a matrix
    target_matrix = [
        np.array([[1., 0., 0., 0.],
                  [0., 1., 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., -1.],
                  [0., 0., 0., 0.],
                  [0., 0., 0., 0.],
                  [0., 0., 0., 0.],
                  [0., 0., 0., 0.],
                  [0., 0., 0., 0.],
                  [0., 0., 0., 0.]]),
        np.array([[1., 0., 0., 0.],
                  [0., 1., 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.],
                  [0., 0., 0., 0.],
                  [0., 0., 0., 0.],
                  [0., 0., 0., 0.],
                  [0., 0., 0., 0.],
                  [0., 0., 0., 0.],
                  [0., 0., 0., 0.]])
    ]

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

    def modes2photons(modes, n_modes):
        photons = [0] * n_modes
        for mode in modes:
            photons[mode] += 1
        return photons

    def photons2modes(photons):
        n_photons = sum(photons)
        modes = []
        for mode in range(len(photons)):
            photon = photons[mode]
            for i in range(photon):
                modes.append(mode)
        return modes

    def distinct(y):
        seen = set()
        return [x for x in y if x not in seen and not seen.add(x)]

    def next_idx(idx, maximum):
        n = len(idx)
        i = -1
        idx[i] += 1
        while idx[i] >= maximum:
            idx[i] = 0
            i -= 1
            if i >= -n:
                idx[i] += 1
            else:
                return False
        return True

    # x = [1, 2, 3, 4, 5]
    # idx = [1, 3]
    # print([x[i] for i in idx])

    # index = [0, 0, 0]
    # n = 2
    # for i in range(10):
    #     print(next_idx(index, n))

    # for m in range(1, 5):
    #     for p in range(1, 3):
    #         args = [list(range(m))] * p
    #         indices = [tuple(sorted(list(i))) for i in product(*args)]
    #         indices = [list(x) for x in distinct(indices)]
    #         # print(len(indices), end=' ')
    #     # print()
    #         ancilla_states = [tuple(photons2modes(x)) for x in [sorted(modes2photons(x, m)) for x in indices]]
    #         ancilla_states = [list(x) for x in distinct(ancilla_states)]
    #         for ac in ancilla_states:
    #             ancilla_state = np.array(ac)
    #
    #             index0 = [0]
    #             index1 = [0]
    #             while True:
    #                 # measurements0 = [item for sublist in [indices[i] for i in index0] for item in sublist]
    #                 measurements0 = [indices[i] for i in index0]
    #                 while True:
    #                     if not set(index0) & set(index1):
    #                     # if not set(index0) & set(index1) and not index0[0] == index0[1] and not index1[0] == index1[1]:
    #                         # measurements1 = [item for sublist in [indices[i] for i in index1] for item in sublist]
    #                         measurements1 = [indices[i] for i in index1]
    #                         measurements = [
    #                             np.array(measurements0),
    #                             np.array(measurements1)
    #                         ]
    #
    #                         print("m = ", m, ", p = ", p, sep='')
    #                         print(ancilla_state)
    #                         print(measurements)
    #                         search = MultiCircuitSearch(target_matrix, input_basic_states, n_ancilla_modes=m,
    #                                                     measurements=measurements,
    #                                                     ancilla_state=ancilla_state,
    #                                                     output_basic_states=output_basic_states,
    #                                                     topology=tl.Parallel, device='cpu')
    #                         circuit = search.run(min_probability=2 / 27, n_epochs=500, print_info=True)
    #
    #                     if not next_idx(index1, len(indices)):
    #                         break
    #                 if not next_idx(index0, len(indices)):
    #                     break

    # Launch the search!
    # circuit = search.run(min_probability=2 / 27, n_epochs=1000, print_info=True)

    ancilla_state = np.array([0, 1])
    measurements = [
        np.array([[0, 1]]),
        np.array([[2, 3]])
    ]

    search = MultiCircuitSearch(target_matrix, input_basic_states, n_ancilla_modes=9,
                                measurements=measurements,
                                ancilla_state=ancilla_state,
                                output_basic_states=output_basic_states,
                                topology=tl.Parallel, device='cuda')
    circuit = search.run(min_probability=2 / 27, n_epochs=800, print_info=True)

    # Print result
    # print("Circuit:")
    # circuit.print()

    # Save result
    # circuit.to_loqc_tech("result.json")
