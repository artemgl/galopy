import torch
from math import pi, factorial
from itertools import product
from abc import ABC, abstractmethod

# Multiply to get angle in radians from int
# TODO: move it out of here
RADIANS = pi / 18000.


class GeneticAlgorithm(ABC):
    def __init__(self, device: str, depth=1, n_state_modes=2, n_ancilla_modes=0, n_state_photons=1, n_ancilla_photons=0,
                 max_success_measurements=1):
        """
        Algorithm searching a scheme
            Parameters:
                device: The device on which you want to store data and perform calculations (e.g. 'cuda')

                depth: Number of local two-mode unitary transforms. One transform contains two phase shifters and one beam splitter

                n_state_modes: Number of modes in which input and output states are

                n_ancilla_modes: Number of modes in which ancilla photons are

                n_state_photons: Number of photons in input and output states

                n_ancilla_photons: Number of ancilla photons

                max_success_measurements: Max count of measurements that we consider as successful gate operation
        """
        self._device = device

        self._depth = depth

        self._n_state_modes = n_state_modes
        self._n_ancilla_modes = n_ancilla_modes
        # Total number of modes in scheme
        self._n_modes = n_state_modes + n_ancilla_modes
        # Number of modes in which unitary transform is performed
        # It's considered that all of ancilla modes always participate in this transform
        self._n_work_modes = self._n_modes

        self._n_state_photons = n_state_photons
        self._n_ancilla_photons = n_ancilla_photons
        # Total number of photons
        self._n_photons = n_state_photons + n_ancilla_photons

        self._max_success_measurements = max_success_measurements

    def __normalize_coeffs(self, population):
        """
        Bring the data to a convenient form.

        First self._depth of parameters are angles for the first rz rotation.

        Next self._depth are angles for the second rz rotation.

        Next self._depth are angles for ry rotation.

        Next 2 * self._depth are pairs of modes on which local unitary is performed.

        Next one is the number of measurements.

        Next self._n_ancilla_photons are initial state for ancilla photons.

        The last are results of measurements.
        """
        population[:, 0:2 * self._depth] %= 36000
        population[:, 2 * self._depth:3 * self._depth] %= 9000  # Not continious in case of mutations

        population[:, 3 * self._depth:5 * self._depth] %= self._n_work_modes
        # If modes are equal, increment the second one
        x = population[:, 3 * self._depth:5 * self._depth:2]
        y = population[:, 3 * self._depth + 1:5 * self._depth:2]
        mask = y == x
        y[mask] += 1
        y[mask] %= self._n_work_modes

        population[:, 5 * self._depth] %= self._max_success_measurements
        population[:, 5 * self._depth + 1:] %= self._n_ancilla_modes

        if self._n_ancilla_photons > 0:
            modes, _ = population[:, 5 * self._depth + 1:].reshape(population.shape[0], -1, self._n_ancilla_photons).sort()
            population[:, 5 * self._depth + 1:] = modes.reshape(population.shape[0], -1)

        return population

    def __gen_random_population(self, n_parents: int):
        """Create a random initial population made of n_parents."""
        res = torch.randint(0, 36000,
                            (n_parents,
                             5 * self._depth +  # 3 angles and 2 modes for each one
                             1 +  # Number of measurements
                             self._n_ancilla_photons * (1 +  # for initial state of ancilla photons
                                                        self._max_success_measurements)),  # results of measurements
                            device=self._device, dtype=torch.int)

        return self.__normalize_coeffs(res)

    def __build_permutation_matrix(self):
        """Create matrix for output state computing."""
        def to_idx(*modes):
            """Convert multi-dimensional index to one-dimensional."""
            res = 0
            for mode in modes:
                res = res * self._n_modes + mode
            return res

        args = [list(range(self._n_modes))] * self._n_photons
        indices = [list(i) for i in product(*args)]

        normalized_indices = [idx.copy() for idx in indices]
        for idx in normalized_indices:
            idx.sort()

        all_indices = list(map(lambda x, y: [to_idx(*x), to_idx(*y)], normalized_indices, indices))
        vals = [1.] * len(all_indices)

        return torch.sparse_coo_tensor(torch.tensor(all_indices).t(), vals, device=self._device, dtype=torch.complex64)

    def __build_normalization_matrix(self, permutation_matrix):
        """
        Create matrices for transforming between two representations of state: Dirac form and operator form.
        It's considered that operator acts on the vacuum state.

            First matrix:  Dirac    -> operator ( |n> -> a^n / sqrt(n!) )

            Second matrix: operator -> Dirac    ( a^n -> sqrt(n!) * |n> )
        """
        vector = torch.ones(permutation_matrix.shape[1], 1, device=self._device, dtype=torch.complex64,
                            requires_grad=False)
        vector = torch.sparse.mm(permutation_matrix, vector).to_sparse_coo()

        indices = vector.indices()[0].reshape(1, -1)
        indices = torch.cat((indices, indices))
        c = factorial(self._n_photons)
        norm_mtx = torch.sparse_coo_tensor(indices, (vector.values() / c).sqrt(),
                                           size=permutation_matrix.shape, device=self._device, requires_grad=False)
        inv_norm_mtx = torch.sparse_coo_tensor(indices, (c / vector.values()).sqrt(),
                                               size=permutation_matrix.shape, device=self._device, requires_grad=False)

        return norm_mtx, inv_norm_mtx

    def __read_scheme_unitary(self, population):
        """Read genome and return the unitary transformation of the scheme it represents."""
        # Masks to get indices of modes
        # TODO: Move to outer scope
        mask_modes_2 = torch.tensor([[3 * self._depth + 2 * i,
                                      3 * self._depth + 2 * i,
                                      3 * self._depth + 2 * i + 1,
                                      3 * self._depth + 2 * i + 1]
                                     for i in range(self._depth)], dtype=torch.long, device=self._device)
        mask_modes_3 = torch.tensor([[3 * self._depth + 2 * i,
                                      3 * self._depth + 2 * i + 1,
                                      3 * self._depth + 2 * i,
                                      3 * self._depth + 2 * i + 1]
                                     for i in range(self._depth)], dtype=torch.long, device=self._device)
        # Indices to slice correctly
        # TODO: Move to outer scope
        indices_0 = torch.tensor([[i] * 4 * self._depth for i in range(population.shape[0])], device=self._device)\
            .reshape(-1, self._depth, 4)
        indices_1 = torch.tensor([[i, i, i, i] for i in range(self._depth)] * population.shape[0], device=self._device)\
            .reshape(-1, self._depth, 4)
        indices_2 = population[:, mask_modes_2]
        indices_3 = population[:, mask_modes_3]

        # Get angles
        phies = population[:, 0:self._depth]
        thetas = population[:, 2 * self._depth:3 * self._depth]
        lambdas = population[:, self._depth:2 * self._depth]

        # Write unitary coefficients for each triple of angles
        unitaries_coeffs = torch.zeros(population.shape[0], self._depth, 4, device=self._device, dtype=torch.complex64)
        unitaries_coeffs[:, :, 0] = torch.cos(thetas * RADIANS)
        unitaries_coeffs[:, :, 2] = torch.sin(thetas * RADIANS)
        unitaries_coeffs[:, :, 1] = -unitaries_coeffs[:, :, 2] * torch.exp(1.j * phies * RADIANS)
        unitaries_coeffs[:, :, 3] = unitaries_coeffs[:, :, 0] * torch.exp(1.j * (phies + lambdas) * RADIANS)
        unitaries_coeffs[:, :, 2] *= torch.exp(1.j * lambdas * RADIANS)

        # Create an unitary for each triple of angles
        unitaries = torch.zeros(population.shape[0], self._depth, self._n_modes, self._n_modes,
                                device=self._device, dtype=torch.complex64)
        # TODO: Move mask to outer scope
        mask = torch.eye(self._n_modes, device=self._device, dtype=torch.bool)
        unitaries[:, :, mask] = 1.
        unitaries[indices_0.long(), indices_1.long(), indices_2.long(), indices_3.long()] = unitaries_coeffs

        # Multiply all the unitaries to get one for the whole scheme
        # TODO: Optimize
        for i in range(1, self._depth):
            unitaries[:, 0] = unitaries[:, i].matmul(unitaries[:, 0])

        return unitaries[:, 0].clone()

    @abstractmethod
    def _fill_state(self, population, state_vector):
        """Assign the initial state in Dirac form."""
        pass

    def __calculate_state(self, population, permutation_matrix, normalization_matrix, inv_normalization_matrix):
        """Express initial state in terms of output birth operators and transform to Dirac form."""
        # TODO: move size and size_mtx to outer scope ?
        # TODO: create state_vector once at the beginning ?
        # Create state vector in operator form
        size = [population.shape[0]] + [self._n_modes] * self._n_photons + [1]
        size_mtx = [population.shape[0]] + [1] * (self._n_photons - 1) + [self._n_modes] * 2
        state_vector = torch.zeros(*size, device=self._device, dtype=torch.complex64)

        # TODO: create vector in method __fill_state()
        state_vector = self._fill_state(population, state_vector)

        # Normalize (transform to operator form)
        state_vector = state_vector.reshape(population.shape[0], -1)
        state_vector.t_()
        state_vector = torch.sparse.mm(normalization_matrix, state_vector)
        state_vector.t_()
        state_vector = state_vector.reshape(*size)

        # Get unitary transforms
        unitaries = self.__read_scheme_unitary(population)
        unitaries = unitaries.reshape(*size_mtx)

        # Apply unitaries to all photons in state
        state_vector = unitaries.matmul(state_vector)
        # TODO: matrix reshape instead of vector transposing ?
        # TODO: Optimize? (Maybe firstly have sparse vector, then convert it to dense before some iteration)
        for i in range(1, self._n_photons):
            state_vector.transpose_(i, self._n_photons)
            state_vector = unitaries.matmul(state_vector)
            state_vector.transpose_(i, self._n_photons)

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

    @abstractmethod
    def _calculate_fidelity_and_probability(self, state_vector):
        pass

    def __calculate_loss(self, state_vector):
        pass

    def __crossover(self, population):
        pass

    def __mutate(self, population):
        pass

    def __select(self, population):
        pass

    def run(self):
        pass
