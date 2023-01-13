import numpy as np
import torch
from galopy.lonet import LoNet
from math import ceil, log


if __name__ == '__main__':
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
    # target_matrix = np.array([[1., 0., 0., 0.],
    #                           [0., 1., 0., 0.],
    #                           [0., 0., 1., 0.],
    #                           [0., 0., 0., -1.]])
    # input_basic_states = np.array([[0, 2],
    #                                [0, 3],
    #                                [1, 2],
    #                                [1, 3]])
    measurements = np.array([[1, 2]])
    ancilla_state = np.array([1, 2])
    net = LoNet(target_matrix, input_basic_states, output_basic_states=output_basic_states,
                measurements=measurements, ancilla_state=ancilla_state, n_ancilla_modes=3)
    net.to_loqc_tech("pip.txt")

