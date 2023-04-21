import torch
from math import tau, pi, factorial
import json
from itertools import product
from galopy.circuit import Circuit


class Population:
    def __init__(self, n_modes, n_ancilla_modes, n_state_photons, bs_angles, ps_angles, topologies,
                 initial_ancilla_state, measurements, device='cpu'):
        self.device = device

        # TODO: проверка согласованности размерностей
        self.n_individuals = bs_angles.shape[0]
        self.depth = bs_angles.shape[1]
        self.n_modes = n_modes
        self.n_work_modes = self.n_modes
        self.n_ancilla_photons = initial_ancilla_state.shape[1]
        self.n_state_photons = n_state_photons
        self.n_photons = self.n_state_photons + self.n_ancilla_photons
        self.n_ancilla_modes = n_ancilla_modes
        self.n_success_measurements = measurements.shape[1]

        self.bs_angles = bs_angles
        self.ps_angles = ps_angles
        self.topologies = topologies
        self.initial_ancilla_states = initial_ancilla_state
        self.measurements = measurements

    def get_parameters(self):
        return (
            self.n_modes,
            self.n_ancilla_modes,
            self.bs_angles,
            self.ps_angles,
            self.topologies,
            self.initial_ancilla_states,
            self.measurements
        )

    def set_precomputed(self, other):
        self._mask_for_unitaries = other._mask_for_unitaries
        self._permutation_matrix = other._permutation_matrix
        self._normalization_matrix = other._normalization_matrix
        self._inverted_normalization_matrix = other._inverted_normalization_matrix

    def precompute_extra(self):
        """Compute auxiliary objects."""
        # Prepare mask for unitaries in method __read_scheme_unitary()
        self._mask_for_unitaries = torch.eye(self.n_modes, device=self.device, dtype=torch.bool)
        self._construct_permutation_matrix()
        self._construct_normalization_matrix(self._permutation_matrix)

    def _construct_permutation_matrix(self):
        """
        The matrix for output state computing.
        Multiply by it state vector to sum up all like terms.
        For example, vector (a0 * a1 + a1 * a0) will become 2 * a0 * a1
        """
        # TODO: возможно через reshape без to_idx ?
        def to_idx(*modes):
            """Convert multi-dimensional index to one-dimensional."""
            res = 0
            for mode in modes:
                res = res * self.n_modes + mode
            return res

        args = [list(range(self.n_modes))] * self.n_photons
        indices = [list(i) for i in product(*args)]

        normalized_indices = [idx.copy() for idx in indices]
        for idx in normalized_indices:
            idx.sort()

        all_indices = list(map(lambda x, y: [to_idx(*x), to_idx(*y)], normalized_indices, indices))
        vals = [1.] * len(all_indices)

        self._permutation_matrix = torch.sparse_coo_tensor(torch.tensor(all_indices).t(), vals, device=self.device,
                                                           dtype=torch.complex64)

    def _construct_normalization_matrix(self, permutation_matrix):
        """
        Get matrices for transforming between two representations of state: Dirac form and operator form.
        It's considered that operator acts on the vacuum state.

            First matrix:  Dirac    -> operator ( |n> -> a^n / sqrt(n!) )

            Second matrix: operator -> Dirac    ( a^n -> sqrt(n!) * |n> )
        """
        vector = torch.ones(permutation_matrix.shape[1], 1, device=self.device, dtype=torch.complex64)
        vector = torch.sparse.mm(permutation_matrix, vector).to_sparse_coo()

        indices = vector.indices()[0].reshape(1, -1)
        indices = torch.cat((indices, indices))
        c = factorial(self.n_photons)

        self._normalization_matrix = torch.sparse_coo_tensor(indices, (vector.values() / c).sqrt(),
                                                             size=permutation_matrix.shape, device=self.device)
        self._inverted_normalization_matrix = torch.sparse_coo_tensor(indices, (c / vector.values()).sqrt(),
                                                                      size=permutation_matrix.shape,
                                                                      device=self.device)

    def _construct_input_state(self, input_states):
        """Get input state in Dirac form."""
        n_input_states = input_states.shape[0]
        n_photons = input_states.shape[1] + self.n_ancilla_photons

        # TODO: move size to outer scope ?
        size = [self.n_individuals, n_input_states] + [self.n_modes] * n_photons + [1]

        # TODO: outer scope ?
        sub_indices_0 = torch.tensor([[i] * n_input_states for i in range(self.n_individuals)],
                                     device=self.device).reshape(-1, 1)
        sub_indices_1 = torch.tensor([list(range(n_input_states)) for _ in range(self.n_individuals)],
                                     device=self.device).reshape(-1, 1)

        # TODO: outer scope ?
        if self.n_ancilla_photons > 0:
            indices = torch.cat((sub_indices_0, sub_indices_1,
                                 self.initial_ancilla_states[sub_indices_0.reshape(-1)]
                                 .reshape(-1, self.n_ancilla_photons),
                                 input_states[sub_indices_1.reshape(-1)],
                                 torch.zeros_like(sub_indices_0, device=self.device)), 1).t()
        else:
            indices = torch.cat((sub_indices_0, sub_indices_1,
                                 input_states[sub_indices_1.reshape(-1)],
                                 torch.zeros_like(sub_indices_0, device=self.device)), 1).t()

        values = torch.ones(indices.shape[1], device=self.device, dtype=torch.complex64)

        return torch.sparse_coo_tensor(indices, values, size=size, device=self.device)

    def _construct_output_state(self, n_input_states, state_vector):
        """Get full output state in Dirac form."""
        # TODO: move size and size_mtx to outer scope ?
        # TODO: create state_vector once at the beginning ?
        size_mtx = [self.n_individuals] + [1] * self.n_photons + [self.n_modes] * 2

        perm_mtx = self._permutation_matrix
        norm_mtx, inv_norm_mtx = self._normalization_matrix, self._inverted_normalization_matrix

        # Create state vector in Dirac form
        vector_shape = state_vector.shape

        # Normalize (transform to operator form)
        state_vector = state_vector.reshape(self.n_individuals * n_input_states, -1)
        state_vector.t_()
        state_vector = torch.sparse.mm(norm_mtx, state_vector)
        state_vector.t_()
        state_vector = state_vector.reshape(vector_shape)

        # Get unitary transforms
        unitaries = self._read_scheme_unitary()
        unitaries = unitaries.reshape(*size_mtx)

        # Apply unitaries to all photons in state
        state_vector = unitaries.matmul(state_vector)
        # TODO: matrix reshape instead of vector transposing ?
        # TODO: Optimize? (Maybe firstly have sparse vector, then convert it to dense before some iteration)
        for i in range(self.n_photons - 1):
            state_vector.transpose_(-3 - i, -2)
            state_vector = unitaries.matmul(state_vector)
            state_vector.transpose_(-3 - i, -2)

        state_vector = state_vector.reshape(self.n_individuals * n_input_states, -1)
        state_vector.t_()
        # TODO: Vector to sparse coo before multiplying
        # Sum up indistinguishable states with precomputed permutation matrix
        state_vector = torch.sparse.mm(perm_mtx, state_vector)
        # Transform to Dirac form
        state_vector = torch.sparse.mm(inv_norm_mtx, state_vector)
        state_vector.t_()
        state_vector = state_vector.reshape(vector_shape)

        return state_vector.to_sparse_coo()

    def _extract_output_states(self, n_input_states, state_vector, output_states):
        """Given the full output vector, get just output sub-vector."""
        n_output_states = output_states.shape[0]

        # TODO: precompute ?
        indices_0 = torch.tensor([[i] * n_input_states * n_output_states *
                                  self.n_success_measurements for i in range(self.n_individuals)],
                                 device=self.device, dtype=torch.long).reshape(-1)
        indices_1 = torch.tensor(list(range(n_input_states))
                                 * n_output_states * self.n_success_measurements * self.n_individuals,
                                 device=self.device, dtype=torch.long).reshape(-1)
        indices_3 = torch.tensor([[i] * n_input_states for i in range(n_output_states)]
                                 * self.n_individuals * self.n_success_measurements,
                                 device=self.device, dtype=torch.long).reshape(-1)

        a = (indices_0, indices_1)
        indices_3 = output_states[indices_3].long().reshape(-1, self.n_state_photons)
        c = tuple(indices_3[:, i] for i in range(self.n_state_photons))

        if self.n_ancilla_photons > 0:
            ancillas = self.measurements.long().reshape(self.n_individuals, self.n_success_measurements, -1)

            indices_2 = torch.tensor([[i] * n_input_states * n_output_states
                                      for i in range(self.n_success_measurements)] * self.n_individuals,
                                     device=self.device, dtype=torch.long).reshape(-1)
            indices_2 = ancillas[indices_0, indices_2].reshape(-1, self.n_ancilla_photons)
            b = tuple(indices_2[:, i] for i in range(self.n_ancilla_photons))
            indices = sum((a, b, c, (0,)), ())
        else:
            indices = sum((a, c, (0,)), ())

        return state_vector.to_dense()[indices].clone().reshape(self.n_individuals, self.n_success_measurements,
                                                                n_output_states, n_input_states)

    def construct_transforms(self, input_states, output_states):
        """
        Construct transforms which are represented by circuits.
        Output matrices have shape (n_output_states, n_input_states).
        So, these matrices show how input states will be represented via superposition of
        output states after gate performing.
        """
        input_state_vector = self._construct_input_state(input_states)
        full_output_state_vector = self._construct_output_state(input_states.shape[0],
                                                                input_state_vector.to_dense())
        return self._extract_output_states(input_states.shape[0],
                                           full_output_state_vector,
                                           output_states)

    def to_file(self, file_name):
        """Write data to file."""
        with open(file_name, 'w') as f:
            f.write(str(self.n_modes))
            f.write("\n")
            f.write(str(self.n_ancilla_modes))
            f.write("\n")
            f.write(str(self.n_state_photons))
            f.write("\n")
            f.write(json.dumps(self.bs_angles.cpu().numpy().tolist()))
            f.write("\n")
            f.write(json.dumps(self.ps_angles.cpu().numpy().tolist()))
            f.write("\n")
            f.write(json.dumps(self.topologies.cpu().numpy().tolist()))
            f.write("\n")
            f.write(json.dumps(self.initial_ancilla_states.cpu().numpy().tolist()))
            f.write("\n")
            f.write(json.dumps(self.measurements.cpu().numpy().tolist()))

    # TODO: при нескольких измерениях не допускать повторяющиеся
    def _normalize_data(self):
        """
        Bring the data to a convenient form.
        """
        self.bs_angles %= tau
        self.ps_angles %= tau

        self.topologies %= self.n_work_modes
        x = self.topologies[..., 0]
        y = self.topologies[..., 1]
        mask = x == y
        y[mask] += 1
        y[mask] %= self.n_work_modes
        self.topologies, _ = self.topologies.sort()

        if self.n_ancilla_modes > 0:
            self.initial_ancilla_states, _ = (self.initial_ancilla_states % self.n_ancilla_modes).sort()
            self.measurements, _ = (self.measurements % self.n_ancilla_modes).sort()

    def _read_scheme_unitary(self):
        """Read genome and return the unitary transformation of the scheme it represents."""
        depth = self.topologies.shape[1]

        # Indices to slice correctly
        # TODO: Move to outer scope
        indices_0 = torch.tensor([[i] * 4 * depth for i in range(self.n_individuals)], device=self.device)\
            .reshape(-1, depth, 4)
        indices_1 = torch.tensor([[i, i, i, i] for i in range(depth)] * self.n_individuals, device=self.device)\
            .reshape(-1, depth, 4)
        indices_2 = self.topologies[..., [0, 0, 1, 1]]
        indices_3 = self.topologies[..., [0, 1, 0, 1]]

        # Get angles
        thetas = self.bs_angles[..., 0]
        phis = self.bs_angles[..., 1]

        # Write unitary coefficients for each beam splitter
        unitaries_coeffs = torch.zeros(self.n_individuals, depth, 4, device=self.device, dtype=torch.complex64)
        unitaries_coeffs[:, :, 0] = torch.cos(thetas)
        unitaries_coeffs[:, :, 1] = torch.exp(-1.j * phis) * torch.sin(thetas)
        unitaries_coeffs[:, :, 2] = -torch.exp(1.j * phis) * torch.sin(thetas)
        unitaries_coeffs[:, :, 3] = torch.cos(thetas)

        # Create an unitary for each beam splitter
        unitaries = torch.zeros(self.n_individuals, depth, self.n_modes, self.n_modes,
                                device=self.device, dtype=torch.complex64)
        unitaries[:, :, self._mask_for_unitaries] = 1.
        unitaries[indices_0.long(), indices_1.long(), indices_2.long(), indices_3.long()] = unitaries_coeffs

        # Multiply all the unitaries to get one for the whole scheme
        # TODO: Optimize
        for i in range(1, depth):
            unitaries[:, 0] = unitaries[:, i].matmul(unitaries[:, 0])

        unitaries_ps = torch.zeros(self.n_individuals, self.n_modes, self.n_modes,
                                   device=self.device, dtype=torch.complex64)
        coeffs = self.ps_angles
        if self.n_work_modes < self.n_modes:
            coeffs = torch.cat((coeffs, torch.zeros(self.n_individuals, self.n_modes - self.n_work_modes,
                                                    device=self.device)), dim=1)
        unitaries_ps[:, self._mask_for_unitaries] = torch.exp(1j * coeffs)

        return unitaries_ps.matmul(unitaries[:, 0])

    # def crossover(self, n_offsprings):
    #     """Get random dads and moms and perform crossover."""
    #     dads = torch.randint(0, self.n_individuals, size=(n_offsprings,), device=self.device)
    #     moms = torch.randint(0, self.n_individuals, size=(n_offsprings,), device=self.device)
    #
    #     mask = torch.rand_like(self._bs_angles[moms, ...], device=self.device) < 0.5
    #     bs_angles = torch.where(mask, self._bs_angles[moms, ...], self._bs_angles[dads, ...])
    #
    #     mask = torch.rand_like(self._ps_angles[moms, ...], device=self.device) < 0.5
    #     ps_angles = torch.where(mask, self._ps_angles[moms, ...], self._ps_angles[dads, ...])
    #
    #     mask = torch.rand_like(self._topologies[moms, ...], device=self.device, dtype=torch.float) < 0.5
    #     topologies = torch.where(mask, self._topologies[moms, ...], self._topologies[dads, ...])
    #
    #     mask = torch.rand_like(self._initial_ancilla_states[moms, ...], device=self.device, dtype=torch.float) < 0.5
    #     initial_ancilla_states = torch.where(mask, self._initial_ancilla_states[moms, ...],
    #                                          self._initial_ancilla_states[dads, ...])
    #
    #     mask = torch.rand_like(self._measurements[moms, ...], device=self.device, dtype=torch.float) < 0.5
    #     measurements = torch.where(mask, self._measurements[moms, ...], self._measurements[dads, ...])
    #
    #     return Population(self._permutation_matrix, self._normalization_matrix, self._inverted_normalization_matrix,
    #                       bs_angles, ps_angles, topologies, initial_ancilla_states, measurements,
    #                       self.n_modes, self.n_ancilla_modes, self.device)
    def crossover(self, n_offsprings):
        """Get random dads and moms and perform crossover."""
        dads = torch.randint(0, self.n_individuals, size=(n_offsprings,), device=self.device)
        moms = torch.randint(0, self.n_individuals, size=(n_offsprings,), device=self.device)

        separators = torch.randint(0, self.depth, size=(n_offsprings, 1), device=self.device)
        mask = torch.zeros(n_offsprings, self.depth, dtype=torch.bool, device=self.device)
        mask.scatter_(dim=1, index=separators, value=True)
        mask = torch.cumsum(mask, dim=1).bool().view(-1, self.depth, 1)

        bs_angles = torch.where(mask, self.bs_angles[moms, ...], self.bs_angles[dads, ...])
        ps_angles = self.ps_angles[moms, ...]
        topologies = torch.where(mask, self.topologies[moms, ...], self.topologies[dads, ...])
        initial_ancilla_states = self.initial_ancilla_states[dads, ...]
        measurements = self.measurements[moms, ...]

        result = Population(self.n_modes, self.n_ancilla_modes, self.n_state_photons, bs_angles, ps_angles, topologies,
                            initial_ancilla_states, measurements, self.device)
        result.set_precomputed(self)

        return result

    def mutate(self, mutation_probability):
        """Mutate individuals."""
        mask = torch.rand_like(self.bs_angles, device=self.device) < mutation_probability
        deltas = torch.rand(size=(mask.sum().item(),), device=self.device) * 0.5 * pi - 0.25 * pi
        self.bs_angles[mask] += deltas

        mask = torch.rand_like(self.ps_angles, device=self.device) < mutation_probability
        deltas = torch.rand(size=(mask.sum().item(),), device=self.device) * 0.5 * pi - 0.25 * pi
        self.ps_angles[mask] += deltas

        mask = torch.rand_like(self.topologies, device=self.device, dtype=torch.float) < mutation_probability
        deltas = torch.randint(0, self.n_work_modes, size=(mask.sum().item(),), device=self.device)
        self.topologies[mask] += deltas

        if self.n_ancilla_modes > 0:
            mask = torch.rand_like(self.initial_ancilla_states, device=self.device, dtype=torch.float) < mutation_probability
            deltas = torch.randint(0, self.n_ancilla_modes, size=(mask.sum().item(),), device=self.device)
            self.initial_ancilla_states[mask] += deltas

            mask = torch.rand_like(self.measurements, device=self.device, dtype=torch.float) < mutation_probability
            deltas = torch.randint(0, self.n_ancilla_modes, size=(mask.sum().item(),), device=self.device)
            self.measurements[mask] += deltas

        self._normalize_data()

    def select(self, fitness, n_to_select):
        """
        Select the given number of individuals from population.
        The choice is based on the fitness.
        """
        fitness, indices = torch.topk(fitness, n_to_select)

        bs_angles = self.bs_angles[indices, ...]
        ps_angles = self.ps_angles[indices, ...]
        topologies = self.topologies[indices, ...]
        initial_ancilla_states = self.initial_ancilla_states[indices, ...]
        measurements = self.measurements[indices, ...]

        result = Population(self.n_modes, self.n_ancilla_modes, self.n_state_photons, bs_angles, ps_angles, topologies,
                            initial_ancilla_states, measurements, self.device)
        result.set_precomputed(self)

        return result, fitness

    def __add__(self, other):
        bs_angles = torch.cat((self.bs_angles, other.bs_angles), 0)
        ps_angles = torch.cat((self.ps_angles, other.ps_angles), 0)
        topologies = torch.cat((self.topologies, other.topologies), 0)
        initial_ancilla_states = torch.cat((self.initial_ancilla_states, other.initial_ancilla_states), 0)
        measurements = torch.cat((self.measurements, other.measurements), 0)

        result = Population(self.n_modes, self.n_ancilla_modes, self.n_state_photons, bs_angles, ps_angles, topologies,
                            initial_ancilla_states, measurements, self.device)
        result.set_precomputed(self)

        return result

    def __getitem__(self, item):
        bs_angles = self.bs_angles[item, ...]
        ps_angles = self.ps_angles[item, ...]
        topologies = self.topologies[item, ...]
        initial_ancilla_states = self.initial_ancilla_states[item, ...]
        measurements = self.measurements[item, ...]

        if isinstance(item, int):
            result = Circuit(self.n_modes, self.n_modes - self.n_ancilla_modes, bs_angles, ps_angles, topologies,
                             initial_ancilla_states, measurements)
        else:
            result = Population(self.n_modes, self.n_ancilla_modes, self.n_state_photons, bs_angles, ps_angles,
                                topologies, initial_ancilla_states, measurements, self.device)
            result.set_precomputed(self)

        return result


class RandomPopulation(Population):
    def __init__(self, n_individuals=1, depth=1, n_modes=2, n_ancilla_modes=0, n_state_photons=0,
                 n_ancilla_photons=0, n_success_measurements=0, device='cpu'):
        bs_angles = torch.rand(n_individuals, depth, 2, device=device) * tau
        ps_angles = torch.rand(n_individuals, n_modes, device=device) * tau

        topologies = torch.randint(0, n_modes, (n_individuals, depth, 2), device=device, dtype=torch.int8)

        if n_ancilla_modes > 0:
            initial_ancilla_states = torch.randint(0, n_ancilla_modes,
                                                   (n_individuals, n_ancilla_photons),
                                                   device=device, dtype=torch.int8)
            measurements = torch.randint(0, n_ancilla_modes,
                                         (n_individuals, n_success_measurements, n_ancilla_photons),
                                         device=device, dtype=torch.int8)
        else:
            initial_ancilla_states = torch.tensor([[]], device=device)
            measurements = torch.tensor([[[]]], device=device)

        super().__init__(n_modes, n_ancilla_modes, n_state_photons, bs_angles, ps_angles, topologies,
                         initial_ancilla_states, measurements, device)
        super().precompute_extra()


class FromFilePopulation(Population):
    def __init__(self, file_name, device='cpu'):
        with open(file_name, 'r') as f:
            n_modes = int(f.readline())
            n_ancilla_modes = int(f.readline())
            n_state_photons = int(f.readline())
            bs_angles = torch.tensor(json.loads(f.readline()), device=device)
            ps_angles = torch.tensor(json.loads(f.readline()), device=device)
            topologies = torch.tensor(json.loads(f.readline()), device=device, dtype=torch.int8)
            initial_ancilla_states = torch.tensor(json.loads(f.readline()), device=device, dtype=torch.int8)
            measurements = torch.tensor(json.loads(f.readline()), device=device, dtype=torch.int8)
        super().__init__(n_modes, n_ancilla_modes, n_state_photons, bs_angles, ps_angles, topologies,
                         initial_ancilla_states, measurements, device)
        super().precompute_extra()
