import numpy as np
import torch
from galopy.lonet import LoNet
from math import ceil, log


def parallel_topology(n_modes):
    res = []
    blocks = [n_modes]
    for i in range(ceil(log(n_modes, 2))):
        # Слой
        blocks = [x for sublist in [[b // 2, b - (b // 2)] for b in blocks] for x in sublist]
        block = []

        # Индекс, с которого начинаются преобразования в общей матрице
        start = 0
        for j in range(len(blocks) // 2):
            # Параллельный блок в слое

            # Количество мод в левой половине
            left = blocks[2 * j]
            # Количество мод в правой половине
            right = blocks[2 * j + 1]
            for k in range(right):
                # Параллельный шаг в блоке
                for m in range(left):
                    # Конкретная мода в левой половине
                    x = start + m
                    y = start + left + (m + k) % right
                    block.append([x, y])
            start += left + right
        res = block + res
    return res


if __name__ == '__main__':
    print(parallel_topology(8))
    exit(0)
    # target_matrix = np.array([[1., 0., 0., 0.],
    #                           [0., 1., 0., 0.],
    #                           [0., 0., 1., 0.],
    #                           [0., 0., 0., -1.],
    #                           [0., 0., 0., 0.],
    #                           [0., 0., 0., 0.],
    #                           [0., 0., 0., 0.],
    #                           [0., 0., 0., 0.],
    #                           [0., 0., 0., 0.],
    #                           [0., 0., 0., 0.]])
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
    target_matrix = np.array([[1., 0., 0., 0.],
                              [0., 1., 0., 0.],
                              [0., 0., 1., 0.],
                              [0., 0., 0., -1.]])
    input_basic_states = np.array([[0, 2],
                                   [0, 3],
                                   [1, 2],
                                   [1, 3]])
    measurements = np.array([[0]])
    ancilla_state = np.array([0])
    net = LoNet(target_matrix, input_basic_states,
                measurements=measurements, ancilla_state=ancilla_state, n_ancilla_modes=1)
    # net = LoNet(target_matrix, input_basic_states, output_basic_states=output_basic_states,
    #             measurements=measurements, ancilla_state=ancilla_state, n_ancilla_modes=4)
    net.to_loqc_tech("pip.txt")

