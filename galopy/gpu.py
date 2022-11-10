import torch
from math import pi, factorial
from itertools import product

# def build_rz(angle):
#     return torch.tensor([[1, 0],
#                          [0, cos(angle) + 1j * sin(angle)]],
#                         device=DEVICE, dtype=torch.complex64)
#
#
# def build_ry(angle):
#     return torch.tensor([[cos(angle / 2), -sin(angle / 2)],
#                          [sin(angle / 2), cos(angle / 2)]],
#                         device=DEVICE, dtype=torch.complex64)
#
#
# def build_unitary(phi, theta, lambd):
#     res = build_rz(lambd)
#     res = res.matmul(build_ry(theta))
#     res = res.matmul(build_rz(phi))
#     return res

if __name__ == "__main__":
    pass
    # res = torch.eye(N_MODES, device=DEVICE, dtype=torch.complex64)

    # pop = gen_random_population(3)

    # pop[:, 5 * DEPTH + 1:5 * DEPTH + 1 + N_ANCILLA_PHOTONS], _ =\
    #     pop[:, 5 * DEPTH + 1:5 * DEPTH + 1 + N_ANCILLA_PHOTONS].sort()
    # print(pop)

    # pop = torch.tensor(
    #     [[30000, 4500, 21000, 2, 1,
    #       12000, 3526, 0, 3, 2,
    #       0, 4500, 0, 3, 1,
    #       1,
    #       2, 1,
    #       1, 2,
    #       4, 2],
    #      [0, 2250, 0, 2, 1,
    #       0, 6553, 18000, 2, 3,
    #       18000, 2250, 18000, 2, 1,
    #       0,
    #       0, 2,
    #       2, 3,
    #       3, 4]], device='cuda:0', dtype=torch.int)

    # pop = torch.tensor(
    #     [[0, 4500, 0, 0, 1,
    #       0, 0, 0, 3, 2,
    #       0, 0, 0, 3, 1,
    #       1,
    #       0, 1,
    #       1, 2,
    #       4, 2],
    #      [0, 6000, 0, 1, 2,
    #       0, 0, 0, 3, 2,
    #       0, 0, 0, 3, 1,
    #       1,
    #       2, 1,
    #       1, 2,
    #       4, 2]
    #      ], device='cuda:0', dtype=torch.int)
    # pop = normalize_coeffs(pop)
    # p = build_permutation_matrix()
    # n, i_n = build_normalization_matrix(p)
    # vr = calculate_state(pop, p, n, i_n)
    # print(vr)

    # p = build_permutation_matrix()
    # print(p)
    # m, inv_m = build_normalization_matrix(p)
    # print(m)
    # print(inv_m)

    # vector = torch.sparse_coo_tensor(indices=torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    #                                                        [3, 3, 4, 4, 4, 4, 3, 3, 4, 4, 4, 4],
    #                                                        [3, 3, 3, 3, 4, 4, 3, 3, 3, 3, 4, 4],
    #                                                        [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2],
    #                                                        [0, 1, 0, 1, 0, 1, 1, 2, 1, 2, 1, 2],
    #                                                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
    #    values=torch.tensor([-0.5000+0.j,  0.5000+0.j, -0.7071+0.j,  0.7071+0.j, -0.5000+0.j,
    #                    0.5000+0.j,  0.3536+0.j,  0.6124+0.j,  0.5000+0.j,  0.8660+0.j,
    #                    0.3536+0.j,  0.6124+0.j]),
    #    device='cuda:0', size=(2, 5, 5, 5, 5, 1))
    # vector = vector.to_dense()
    # vector.transpose_(1, N_PHOTONS)
    # print(vector.to_sparse_coo())

    # vtr = torch.zeros(3, 2, 2, 2, 2, 1, device=DEVICE)
    # vtr[0][0] = 0.5
    # vtr[1][1] = -0.5
    #
    # mtx = torch.tensor([[[1 / sqrt(2.), 1 / sqrt(2.)], [1 / sqrt(2.), -1 / sqrt(2.)]],
    #                     [[0, 1], [1, 0]],
    #                     [[1, 2], [3, 4]]], device=DEVICE)
    # mtx = mtx.reshape(3, 1, 1, 1, 2, 2)
    # # mtx = torch.sparse_coo_tensor(torch.tensor([[0, 1, 1, 3], [0, 1, 2, 3]]), [1.] * 4, device=DEVICE)
    #
    # print(mtx)
    # print(vtr)
    # vtr = mtx.matmul(vtr)
    # vtr[0] = mtx[0].matmul(vtr[0])
    # print(vtr)
    # vtr.transpose_(0, 1)
    # vtr = mtx.matmul(vtr)
    # vtr.transpose_(0, 1)
    # print(vtr)

    # print(torch.sparse.mm(mtx, vtr))

    # population = gen_random_population(2)
    # calculate_state(population)

    # population = torch.tensor(
    #     [[15000,  9000, 10500,     1,     2,
    #        6000,  7053,     0,     0,     1,
    #           0,  9000,     0,     0,     2,
    #           1,
    #           2,     3,     4,
    #           1,     2,     2,
    #           4,     2,     2],
    #      [9000, 9000,  0,     0,     1,
    #          0,    0,  0,     0,     1,
    #       9000, 9000,  0,     2,     3,
    #           0,
    #           3,     4,     2,
    #           3,     3,     4,
    #           0,     4,     0]], device='cuda:0', dtype=torch.int16)
    # print(read_scheme_unitary(population))

    # a = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    # b = torch.tensor([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])
    # print(a[:].matmul(b[:]))

    # # Взять дважды один и тот же индекс
    # # mask = torch.tensor([0, 0], device=DEVICE)
    #
    # indices_0 = torch.tensor([0, 0], device=DEVICE)
    # indices_1 = torch.tensor([0, 1], device=DEVICE)
    # # indices_2 = torch.tensor([0, 1], device=DEVICE)
    # mask = torch.eye(N_MODES, device=DEVICE, dtype=torch.bool)
    # print(unitaries[indices_0, indices_1][:, mask])

    # x = torch.zeros(4, 3, 3, device=DEVICE, dtype=torch.int16)
    # mask = torch.tensor([[[True, False, True], [False, False, False], [True, False, True]]] * 4, device=DEVICE)
    # mask = torch.tensor([[0, 1], [1, 0]], device=DEVICE, dtype=torch.long)
    # x[mask] = torch.tensor([42, 7], device=DEVICE, dtype=torch.int16)
    # indices0 = torch.tensor([0, 0, 1, 0], device=DEVICE)
    # indices1 = torch.tensor([[0, 0, 2, 2], [0, 0, 1, 1]], device=DEVICE)
    # indices2 = torch.tensor([[0, 2, 0, 2], [0, 1, 0, 1]], device=DEVICE)
    # values = torch.tensor([[7, 42, 73, 29], [1, 2, 3, 4]], device=DEVICE, dtype=torch.int16)
    # x.index_put_(indices=(indices0, indices1, indices2), values=values)

    # unitary = build_unitary(phi, theta, lambd)
    # print(unitary)
    # unitary.resize_(4)
    # indices = [[mode0, mode0, mode1, mode1], [mode0, mode1, mode0, mode1]]
    # mtx = torch.sparse_coo_tensor(torch.tensor(indices), unitary, size=(4, 4), device=DEVICE)
    # print(mtx)
    #
    # print(mtx.shape)
    # print(res.shape)
    #
    # res = torch.sparse.mm(mtx, res)
    # print(res)

    # t = torch.tensor([10472])
    # d = torch.exp(1j * t / 10000)
    # print(t)
    # print(d)
    # print(d.dtype)

    # mtx = build_permutation_matrix()
    # indices = [[3, 2, 0], [0, 3, 2]]
    # indices = [to_idx(*i) for i in indices]
    # indices = [[i, 0] for i in indices]
    # vals = [1. / sqrt(len(indices))] * len(indices)
    # vtr = torch.sparse_coo_tensor(torch.tensor(indices).t(), vals, size=(N_MODES ** N_PHOTONS, 1), device=DEVICE)
    # # print(mtx.shape)
    # # print(vtr.shape)
    # print(torch.sparse.mm(mtx, vtr))

    # mtx = mtx.to_dense()
    # size = pow(n_modes, n_photons)
    # mtx = mtx.reshape((size, size))
    # print(mtx)

    # X_train = torch.FloatTensor([0., 1., 2.])
    # print(X_train.is_cuda)
    # X_train = X_train.to(DEVICE)
    # print(X_train.is_cuda)
