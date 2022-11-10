import torch
from math import pi, factorial
from itertools import product

DEVICE = 'cuda'

DEPTH = 3

N_ANCILLA_MODES = 3
# One mode for state, one for total photon count balance
N_MODES_REAL = N_ANCILLA_MODES + 2
# One mode for state only
N_MODES = N_MODES_REAL - 1

N_ANCILLA_PHOTONS = 2
# Two photons for state
N_PHOTONS = N_ANCILLA_PHOTONS + 2

MAX_SUCCESS_MEASUREMENTS = 2

# Multiply to get angle in radians from int
RADIANS = pi / 18000.


def normalize_coeffs(population):
    """Bring the data to a convenient form."""
    population[:, :5 * DEPTH:5] %= 36000
    population[:, 1:5 * DEPTH:5] %= 9000  # Not continious in case of mutations
    population[:, 2:5 * DEPTH:5] %= 36000
    population[:, 3:5 * DEPTH:5] %= N_MODES
    population[:, 4:5 * DEPTH:5] %= N_MODES
    population[:, 5 * DEPTH] %= MAX_SUCCESS_MEASUREMENTS
    population[:, 5 * DEPTH + 1:] %= N_ANCILLA_MODES

    modes, _ = population[:, 5 * DEPTH + 1:].reshape(population.shape[0], -1, N_ANCILLA_PHOTONS).sort()
    population[:, 5 * DEPTH + 1:] = modes.reshape(population.shape[0], -1)

    return population


def gen_random_population(n_parents: int):
    """Create a random initial population made of n_parents."""
    res = torch.randint(-18000, 18000, (n_parents, 5 * DEPTH + 1 + N_ANCILLA_PHOTONS * (MAX_SUCCESS_MEASUREMENTS + 1)),
                        device=DEVICE, dtype=torch.int)

    # TODO: modes can be equal, it's bad
    return normalize_coeffs(res)


def to_idx(*modes):
    """Convert multi-dimensional index to one-dimensional."""
    res = 0
    for mode in modes:
        res = res * N_MODES_REAL + mode
    return res


def build_permutation_matrix():
    """Create matrix for output state computing."""
    args = [list(range(N_MODES_REAL))] * N_PHOTONS
    indices = [list(i) for i in product(*args)]

    normalized_indices = [idx.copy() for idx in indices]
    for idx in normalized_indices:
        idx.sort()

    all_indices = list(map(lambda x, y: [to_idx(*x), to_idx(*y)], normalized_indices, indices))
    vals = [1.] * len(all_indices)

    return torch.sparse_coo_tensor(torch.tensor(all_indices).t(), vals, device=DEVICE, dtype=torch.complex64)


def build_normalization_matrix(permutation_matrix):
    """
    Create matrices for transforming between two representations of state: Dirac form and operator form.
    It's considered that operator acts on the vacuum state.

        First matrix:  Dirac    -> operator ( |n> -> a^n / sqrt(n!) )

        Second matrix: operator -> Dirac    ( a^n -> sqrt(n!) * |n> )
    """
    vector = torch.ones(permutation_matrix.shape[1], 1, device=DEVICE, dtype=torch.complex64, requires_grad=False)
    vector = torch.sparse.mm(permutation_matrix, vector).to_sparse_coo()

    indices = vector.indices()[0].reshape(1, -1)
    indices = torch.cat((indices, indices))
    c = factorial(N_PHOTONS)
    norm_mtx = torch.sparse_coo_tensor(indices, (vector.values() / c).sqrt(),
                                       size=permutation_matrix.shape, device=DEVICE, requires_grad=False)
    inv_norm_mtx = torch.sparse_coo_tensor(indices, (c / vector.values()).sqrt(),
                                           size=permutation_matrix.shape, device=DEVICE, requires_grad=False)

    return norm_mtx, inv_norm_mtx


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


def read_scheme_unitary(population):
    """Read genome and return the unitary transformation of the scheme it represents."""
    # Masks to get indices of modes
    # TODO: Move to outer scope
    mask_modes_2 = torch.tensor([[5 * i + 3,
                                  5 * i + 3,
                                  5 * i + 4,
                                  5 * i + 4]
                                 for i in range(DEPTH)], device=DEVICE)
    mask_modes_3 = torch.tensor([[5 * i + 3,
                                  5 * i + 4,
                                  5 * i + 3,
                                  5 * i + 4]
                                 for i in range(DEPTH)], device=DEVICE)
    # Indices to slice correctly
    # TODO: Move to outer scope
    indices_0 = torch.tensor([[i] * 4 * DEPTH for i in range(population.shape[0])], device=DEVICE).reshape(-1, DEPTH, 4)
    indices_1 = torch.tensor([[i, i, i, i] for i in range(DEPTH)] * population.shape[0], device=DEVICE).reshape(-1,
                                                                                                                DEPTH,
                                                                                                                4)
    indices_2 = population[:, mask_modes_2]
    indices_3 = population[:, mask_modes_3]

    # Get angles
    operators = population[:, :5 * DEPTH]
    phies = operators[:, ::5]
    thetas = operators[:, 1::5]
    lambdas = operators[:, 2::5]

    # Write unitary coefficients for each triple of angles
    unitaries_coeffs = torch.zeros(population.shape[0], DEPTH, 4, device=DEVICE, dtype=torch.complex64)
    unitaries_coeffs[:, :, 0] = torch.cos(thetas * RADIANS)
    unitaries_coeffs[:, :, 2] = torch.sin(thetas * RADIANS)
    unitaries_coeffs[:, :, 1] = -unitaries_coeffs[:, :, 2] * torch.exp(1.j * phies * RADIANS)
    unitaries_coeffs[:, :, 3] = unitaries_coeffs[:, :, 0] * torch.exp(1.j * (phies + lambdas) * RADIANS)
    unitaries_coeffs[:, :, 2] *= torch.exp(1.j * lambdas * RADIANS)

    # Create an unitary for each triple of angles
    unitaries = torch.zeros(population.shape[0], DEPTH, N_MODES_REAL, N_MODES_REAL, device=DEVICE,
                            dtype=torch.complex64)
    # TODO: Move mask to outer scope
    mask = torch.eye(N_MODES_REAL, device=DEVICE, dtype=torch.bool)
    unitaries[:, :, mask] = 1.
    unitaries[indices_0.long(), indices_1.long(), indices_2.long(), indices_3.long()] = unitaries_coeffs

    # Multiply all the unitaries to get one for the whole scheme
    # TODO: Optimize
    for i in range(1, DEPTH):
        unitaries[:, 0] = unitaries[:, i].matmul(unitaries[:, 0])

    return unitaries[:, 0].clone()


def fill_state(population, state_vector):
    """Assign the initial state in Dirac form."""
    # Read initial state for ancilla photons
    # TODO: Optimize
    ancilla_photons = population[:, 5 * DEPTH + 1:5 * DEPTH + 1 + N_ANCILLA_PHOTONS].clone()

    # Fill the state for NSx search
    # TODO: indices move to outer scope ?
    indices = torch.tensor([[i] * N_ANCILLA_PHOTONS for i in range(state_vector.shape[0])], device=DEVICE).t()
    ancilla_idx = [ancilla_photons[:, i].long() for i in range(N_ANCILLA_PHOTONS)]
    print(ancilla_idx)
    idx = [indices] + ancilla_idx + [N_MODES_REAL - 1, N_MODES_REAL - 1]
    state_vector[idx] = 1.
    idx = [indices] + ancilla_idx + [N_MODES_REAL - 2, N_MODES_REAL - 1]
    state_vector[idx] = 1.
    idx = [indices] + ancilla_idx + [N_MODES_REAL - 2, N_MODES_REAL - 2]
    state_vector[idx] = 1.

    return state_vector


def calculate_state(population, permutation_matrix, normalization_matrix, inv_normalization_matrix):
    """Express initial state in terms of output birth operators and transform to Dirac form."""
    # TODO: move size and size_mtx to outer scope ?
    # TODO: create state_vector once at the beginning ?
    # Create state vector in operator form
    size = [population.shape[0]] + [N_MODES_REAL] * N_PHOTONS + [1]
    size_mtx = [population.shape[0]] + [1] * (N_PHOTONS - 1) + [N_MODES_REAL] * 2
    state_vector = torch.zeros(*size, device=DEVICE, dtype=torch.complex64)

    state_vector = fill_state(population, state_vector)

    # Normalize (transform to operator form)
    state_vector = state_vector.reshape(population.shape[0], -1)
    state_vector.t_()
    state_vector = torch.sparse.mm(normalization_matrix, state_vector)
    state_vector.t_()
    state_vector = state_vector.reshape(*size)

    # Get unitary transforms
    unitaries = read_scheme_unitary(population)
    unitaries = unitaries.reshape(*size_mtx)

    # Apply unitaries to all photons in state
    state_vector = unitaries.matmul(state_vector)
    # TODO: Optimize? (Maybe firstly have sparse vector, then convert it to dense before some iteration)
    for i in range(1, N_PHOTONS):
        state_vector.transpose_(i, N_PHOTONS)
        state_vector = unitaries.matmul(state_vector)
        state_vector.transpose_(i, N_PHOTONS)

    state_vector = state_vector.reshape(population.shape[0], -1)
    state_vector.t_()
    # TODO: Vector to sparse coo before multiplying
    # Sum up indistinguishable states with precomputed permutation matrix
    state_vector = torch.sparse.mm(permutation_matrix, state_vector)
    # Transform to Dirac form
    state_vector = torch.sparse.mm(inv_normalization_matrix, state_vector)
    state_vector.t_()
    state_vector = state_vector.reshape(*size)

    return state_vector.to_sparse_coo()


def calculate_fidelity(state_vector):
    pass


if __name__ == "__main__":
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

    pop = torch.tensor(
        [[0, 4500, 0, 0, 1,
          0, 0, 0, 3, 2,
          0, 0, 0, 3, 1,
          1,
          0, 1,
          1, 2,
          4, 2],
         [0, 6000, 0, 1, 2,
          0, 0, 0, 3, 2,
          0, 0, 0, 3, 1,
          1,
          2, 1,
          1, 2,
          4, 2]
         ], device='cuda:0', dtype=torch.int)
    pop = normalize_coeffs(pop)
    p = build_permutation_matrix()
    n, i_n = build_normalization_matrix(p)
    vr = calculate_state(pop, p, n, i_n)
    print(vr)

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
