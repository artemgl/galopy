import numpy as np
from math import tau
import torch
import json
# from galopy.circuit_search import *
from galopy.random_population import RandomPopulation
from galopy.population import from_file, random


if __name__ == "__main__":
    a = torch.tensor([[[[1, 2],
                        [3, 4]],
                       [[5, 6],
                        [7, 8]]],
                      [[[9, 10],
                        [11, 12]],
                       [[13, 14],
                        [15, 16]]]], dtype=torch.int)
    n_modes = 8
    topology = torch.tensor([[[0, 1], [2, 3]], [[5, 4], [7, 6]]])

    # pop = RandomPopulation(None, None, None, n_ancilla_photons=1, n_success_measurements=1)
    # pop = random(None, None, None, n_ancilla_photons=1, n_success_measurements=1)
    # pop.to_file("frog.json")
    # pop.print()

    pop = from_file("frog.json", None, None, None)
    pop.print()

    # t, _ = topology.sort()
    # print(t)

    # b = torch.zeros(2, 2, n_modes, 2, dtype=torch.int)
    # b = torch.zeros(2, 2, 2, n_modes, dtype=torch.int)
    # b = torch.arange(start=0, end=2 * 2 * 2 * n_modes, step=1, dtype=torch.int).view(2, 2, 2, n_modes)
    # b = torch.arange(start=0, end=2 * 2 * 2 * n_modes, step=1, dtype=torch.int).view(2, 2, n_modes, 2)
    # print(b[..., [[0, 1], [2, 3]]])
    # print(b[..., [[0, 1], [2, 3]], :])
    # b[..., topology.reshape(-1)] = a
    # b = b.view(-1, 2)
    # print(a)
    # print(b.index_select(-2, topology.view(-1)))
    # print(b.scatter(-2, topology.view(2, 2, 2, 1), a))
    # print(b.gather(-2, topology.view(2, 2, 1, 2)))
    # print(topology.shape)
    # print(a.shape)

    # x = torch.tensor([[1, 2],
    #                   [3, 4],
    #                   [5, 6]])
    # p = torch.tensor([[0, 1], [1, 0]])
    # y = x.gather(1, p)
    # p = torch.tensor([[0, 0], [2, 2]])
    # y = x.gather(0, p)
    # print(y)
    # y = 0
    # print(x)
