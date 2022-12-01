from galopy.lonet import *
import torch
import numpy as np
from math import pi, sqrt, cos, sin
import matplotlib.pyplot as plt
import copy


def print_circuit(alphas, betas, gammas):
    alphas = (alphas.data * 180. / pi).reshape(-1)
    betas = (betas.data * 180. / pi).reshape(-1)
    gammas = (gammas.data * 180. / pi).reshape(-1)
    print("Alphas:")
    for i in alphas.cpu().numpy().tolist():
        print(f"{i:.6f}")
    print("Betas:")
    for i in betas.cpu().numpy().tolist():
        print(f"{i:.6f}")
    print("Gammas:")
    for i in gammas.cpu().numpy().tolist():
        print(f"{i:.6f}")
    print()


def fidelity(pred, target):
    m = torch.matmul(target.transpose(0, 1).conj(), pred)
    res = torch.matmul(m, m.transpose(0, 1).conj()).trace() + abs(m.trace()) ** 2
    res = torch.abs(res)
    res /= target.shape[1] * (target.shape[1] + 1)
    return res


def run_method(net, loss_fn, optimizer, epochs):
    f_history = []
    p_history = []

    for epoch_index in range(epochs):
        optimizer.zero_grad()

        f, p = net.forward()
        loss_value = loss_fn(f, p)
        # fn_history.append(loss_value.data)
        f_history.append(f[0].item())
        p_history.append(p[0].item())
        loss_value.backward()
        optimizer.step()

    return f_history, p_history


def loss(pred, target):
    m = torch.matmul(target.transpose(0, 1).conj(), pred)
    res = torch.matmul(m, m.transpose(0, 1).conj()).trace() + abs(m.trace()) ** 2
    res = torch.abs(res)
    res /= target.shape[1] * (target.shape[1] + 1)
    return 1. - res


if __name__ == '__main__':

    device = 'cuda:0'
    # device = 'cpu'
    epochs = 1000

    target_matrix = np.array([[1., 0., 0., 0.],
                              [0., 1., 0., 0.],
                              [0., 0., 1., 0.],
                              [0., 0., 0., -1.]])
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

    # a = 1. / sqrt(3.)
    # b = sqrt(2. / 3.)
    # target_matrix = np.array([[a, 0., b, 0., 0., 0.],
    #                           [0., a, 0., 0., b, 0.],
    #                           [-b, 0., a, 0., 0., 0.],
    #                           [0., 0., 0., a, 0., b],
    #                           [0., -b, 0., 0., a, 0.],
    #                           [0., 0., 0., -b, 0., a]])

    # target_matrix = np.array([[0.5,   0.5,  0.5,   0.5],
    #                           [0.5,  0.5j, -0.5, -0.5j],
    #                           [0.5,  -0.5,  0.5,  -0.5],
    #                           [0.5, -0.5j, -0.5,  0.5j]])
    # target_matrix = np.array([[0.5j, -0.5, -0.5j, 0.5],
    #                           [-0.5, 0.5, -0.5, 0.5],
    #                           [-0.5j, -0.5, 0.5j, 0.5],
    #                           [0.5, 0.5, 0.5, 0.5]])
    # target_matrix = np.array([[-0.5 / sqrt(3.) + 0.5j, -0.5 / sqrt(3.) - 0.5j, 1. / sqrt(3.)],
    #                           [-0.5 / sqrt(3.) - 0.5j, -0.5 / sqrt(3.) + 0.5j, 1. / sqrt(3.)],
    #                           [1. / sqrt(3.), 1. / sqrt(3.), 1. / sqrt(3.)]])
    # target_matrix = np.array([[-1. / sqrt(2.), 1. / sqrt(2.)],
    #                           [1. / sqrt(2.), 1. / sqrt(2.)]])
    # target_matrix = np.flip(target_matrix, (0, 1))  # Повернуть на 90 градусов
    # target_matrix = torch.tensor(target_matrix, dtype=torch.complex64, device=device)
    # target_matrix = torch.flip(target_matrix, (0, 1))  # Повернуть на 90 градусов

    # input_basic_states = np.array([[0], [1], [2], [3], [4], [5]])

    input_basic_states = np.array([[0, 2],
                                   [0, 3],
                                   [1, 2],
                                   [1, 3]])
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

    # loss = torch.nn.L1Loss()

    # print_circuit(net.alphas.weight, net.betas.weight, net.gammas.weight)

    # # Проверка осуществляется вызовом кода:
    # def metric(pred, target):
    #     return (pred - target).abs().mean()
    #
    # print(metric(net.forward(), target_matrix).item())
    # # print(net.forward())
    #
    # print_circuit(net.alphas.weight, net.betas.weight, net.gammas.weight)

    plt.figure(figsize=(12, 7))

    # measurements = np.array([[1, 3]])
    # ancilla_state = np.array([1, 3])
    # net = LoNet(target_matrix, input_basic_states, device=device, n_ancilla_modes=4,
    #             measurements=measurements, ancilla_state=ancilla_state, output_basic_states=output_basic_states)
    net = LoNet(target_matrix, input_basic_states, device=device, n_ancilla_modes=2)
    # net = LoNet(target_matrix, input_basic_states, device=device)

    # net_copy = copy.deepcopy(net)
    # net_copy.to(device)
    # optimizer = torch.optim.Adam(net_copy.parameters(), lr=0.01)  # maximize
    # fn_history = run_method(net_copy, target_matrix, loss, optimizer)
    # plt.plot(fn_history, label="1-Fidelity, Adam")

    net_copy = net
    # net_copy = copy.deepcopy(net)
    # net_copy.to(device)
    optimizer = torch.optim.Adam(net_copy.parameters(), lr=0.1, maximize=True)
    # optimizer = torch.optim.SGD(net_copy.parameters(), lr=0.01, nesterov=True, momentum=1., maximize=True)
    f_history, p_history = run_method(net_copy, net_copy.loss2, optimizer, epochs)
    plt.plot(f_history, label="Fidelity")
    plt.plot(p_history, label="Probability")

    # def loss(pred, target):
    #     m = torch.matmul(target.transpose(0, 1).conj(), pred)
    #     res = torch.matmul(m, m.transpose(0, 1).conj()).trace() + abs(m.trace()) ** 2
    #     res = torch.abs(res)
    #     res /= target.shape[1] * (target.shape[1] + 1)
    #     return -torch.log(res)
    #
    # net_copy = copy.deepcopy(net)
    # optimizer = torch.optim.Adam(net_copy.parameters(), lr=0.01)
    # fn_history = run_method(net_copy, target_matrix, loss, optimizer)
    # plt.plot(fn_history, label="-log(Fidelity)")
    #
    # net_copy = copy.deepcopy(net)
    # optimizer = torch.optim.Adam(net_copy.parameters(), lr=0.01)
    # fn_history = run_method(net_copy, target_matrix, torch.nn.L1Loss(), optimizer)
    # plt.plot(fn_history, label="L1 fidelity")

    print_circuit(net_copy.alphas.weight, net_copy.betas.weight, net_copy.gammas.weight)
    fid, prob = net_copy.forward()
    print()
    print("Fidelity:", fid.data)
    print("Probability:", prob.data)

    plt.legend(loc='center right')
    plt.xlabel('epoch')
    plt.ylabel('fidelity')
    plt.grid(linestyle='--', alpha=0.5)
    plt.show()
