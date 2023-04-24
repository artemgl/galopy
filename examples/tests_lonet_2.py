from galopy.gd.lonet import *
import torch
import numpy as np
from math import pi, sqrt
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
    best_circuit = copy.deepcopy(net)
    best_f, best_p = best_circuit.forward()
    best_f = best_f.data
    best_p = best_p.data

    for epoch_index in range(epochs):
        optimizer.zero_grad()

        f, p = net.forward()

        if abs(f - 1.) < 0.001:
            if p > best_p:
                best_circuit = copy.deepcopy(net)
                best_f = f
                best_p = p
        else:
            if f > best_f:
                best_circuit = copy.deepcopy(net)
                best_f = f
                best_p = p

        loss_value = loss_fn(f, p)
        # fn_history.append(loss_value.data)
        f_history.append(f[0].item())
        p_history.append(p[0].item())
        loss_value.backward()
        optimizer.step()

    return best_circuit, f_history, p_history


def loss(pred, target):
    m = torch.matmul(target.transpose(0, 1).conj(), pred)
    res = torch.matmul(m, m.transpose(0, 1).conj()).trace() + abs(m.trace()) ** 2
    res = torch.abs(res)
    res /= target.shape[1] * (target.shape[1] + 1)
    return 1. - res


if __name__ == '__main__':

    # device = 'cuda:0'
    device = 'cpu'
    epochs = 1000

    target_matrix = np.array([[1. / sqrt(2.)],
                              [0.],
                              [0.],
                              [1. / sqrt(2.)],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.],
                              [0.]])

    input_basic_states = np.array([[1, 3]])
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

    plt.figure(figsize=(12, 7))

    measurements = np.array([[1, 1]])
    ancilla_state = np.array([0, 1])
    net = LoNet(target_matrix, input_basic_states, device=device, n_ancilla_modes=2,
                measurements=measurements, ancilla_state=ancilla_state, output_basic_states=output_basic_states)

    net_copy = net
    # net_copy = copy.deepcopy(net)
    # net_copy.to(device)
    optimizer = torch.optim.Adam(net_copy.parameters(), lr=0.01, maximize=True)
    # optimizer = torch.optim.SGD(net_copy.parameters(), lr=0.01, nesterov=True, momentum=0.95, maximize=True)
    best_circuit, f_history, p_history = run_method(net_copy, net_copy.loss2, optimizer, epochs)
    plt.plot(f_history, label="Fidelity")
    plt.plot(p_history, label="Probability")

    net_copy = best_circuit
    net_copy.to_loqc_tech("bell.txt")
    bs_angles = net_copy.bs_angles.weight.data.view(-1, 2)
    print_circuit(bs_angles[:, 0], bs_angles[:, 1], net_copy.ps_angles.weight.data.view(-1))
    fid, prob = net_copy.forward()
    print()
    print("Fidelity:", fid.data)
    print("Probability:", prob.data)

    plt.legend(loc='center right')
    plt.xlabel('epoch')
    plt.ylabel('fidelity')
    plt.grid(linestyle='--', alpha=0.5)
    plt.show()
