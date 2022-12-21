import torch
import pandas as pd
from math import tau, pi
import json


class RealPopulation:
    def __init__(self, perm_mtx, norm_mtx, inv_norm_mtx,
                 bs_angles, topologies, initial_ancilla_states, measurements,
                 n_modes, n_ancilla_modes,
                 device='cpu'):
        self._permutation_matrix = perm_mtx
        self._normalization_matrix = norm_mtx
        self._inverted_normalization_matrix = inv_norm_mtx
        self.device = device

        # TODO: проверка согласованности размерностей
        self.n_individuals = bs_angles.shape[0]
        self.depth = bs_angles.shape[1]
        self.n_modes = n_modes
        self.n_work_modes = self.n_modes
        self.n_ancilla_photons = initial_ancilla_states.shape[1]
        self.n_ancilla_modes = n_ancilla_modes
        self.n_success_measurements = measurements.shape[1]

        self._bs_angles = bs_angles
        self._topologies = topologies
        self._initial_ancilla_states = initial_ancilla_states
        self._measurements = measurements
        self._normalize_data()

        self._precompute_extra()

    def print(self, i):
        angles = self._bs_angles[i].view(-1).cpu().numpy().tolist()
        angles = [f"{180. * angle / pi:.2f}" for angle in angles]

        topology = self._topologies[i].cpu().numpy().tolist()
        topology = [f"{sublist[0]}, {sublist[1]}" for sublist in topology]

        elements = pd.DataFrame({'Element': ['Beam splitter'] * self.depth,
                                 'Angles': angles,
                                 'Modes': topology})

        if self.n_ancilla_photons > 0:
            modes_in = self._initial_ancilla_states[i].view(-1).cpu().numpy()
            # TODO: print all the measurements
            modes_out = self._measurements[i][0].cpu().numpy()

            ancillas = pd.DataFrame({'Mode in': modes_in,
                                     'Mode out': modes_out})
            ancillas.index.name = 'Ancilla photon'

            print(elements, ancillas, sep='\n')
        else:
            print(elements)

    def to_file(self, file_name):
        """Write data to file."""
        with open(file_name, 'w') as f:
            f.write(str(self.n_modes))
            f.write("\n")
            f.write(str(self.n_ancilla_modes))
            f.write("\n")
            f.write(json.dumps(self._bs_angles.cpu().numpy().tolist()))
            f.write("\n")
            f.write(json.dumps(self._topologies.cpu().numpy().tolist()))
            f.write("\n")
            f.write(json.dumps(self._initial_ancilla_states.cpu().numpy().tolist()))
            f.write("\n")
            f.write(json.dumps(self._measurements.cpu().numpy().tolist()))

    # TODO: при нескольких измерениях не допускать повторяющиеся
    def _normalize_data(self):
        """
        Bring the data to a convenient form.
        """
        self._bs_angles %= tau

        self._topologies %= self.n_work_modes
        # self._topologies[..., 1] = torch.where(self._topologies[..., 0] == self._topologies[..., 1],
        #                                        (1 + self._topologies[..., 1]) % self.n_work_modes,
        #                                        self._topologies[..., 1])
        x = self._topologies[..., 0]
        y = self._topologies[..., 1]
        mask = x == y
        y[mask] += 1
        y[mask] %= self.n_work_modes
        self._topologies, _ = self._topologies.sort()

        if self.n_ancilla_modes > 0:
            self._initial_ancilla_states, _ = (self._initial_ancilla_states % self.n_ancilla_modes).sort()
            self._measurements, _ = (self._measurements % self.n_ancilla_modes).sort()

    def _precompute_extra(self):
        """Compute auxiliary objects."""
        # Prepare mask for unitaries in method __read_scheme_unitary()
        self._mask_for_unitaries = torch.eye(self.n_modes, device=self.device, dtype=torch.bool)

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
                                 self._initial_ancilla_states[sub_indices_0.reshape(-1)]
                                 .reshape(-1, self.n_ancilla_photons),
                                 input_states[sub_indices_1.reshape(-1)],
                                 torch.zeros_like(sub_indices_0, device=self.device)), 1).t()
        else:
            indices = torch.cat((sub_indices_0, sub_indices_1,
                                 input_states[sub_indices_1.reshape(-1)],
                                 torch.zeros_like(sub_indices_0, device=self.device)), 1).t()

        values = torch.ones(indices.shape[1], device=self.device, dtype=torch.complex64)

        return torch.sparse_coo_tensor(indices, values, size=size, device=self.device)

    def _read_scheme_unitary(self):
        """Read genome and return the unitary transformation of the scheme it represents."""
        depth = self._topologies.shape[1]

        # Indices to slice correctly
        # TODO: Move to outer scope
        indices_0 = torch.tensor([[i] * 4 * depth for i in range(self.n_individuals)], device=self.device)\
            .reshape(-1, depth, 4)
        indices_1 = torch.tensor([[i, i, i, i] for i in range(depth)] * self.n_individuals, device=self.device)\
            .reshape(-1, depth, 4)
        indices_2 = self._topologies[..., [0, 0, 1, 1]]
        indices_3 = self._topologies[..., [0, 1, 0, 1]]

        # Get angles
        thetas = self._bs_angles

        # Write unitary coefficients for each beam splitter
        unitaries_coeffs = torch.zeros(self.n_individuals, depth, 4, device=self.device, dtype=torch.complex64)
        unitaries_coeffs[:, :, 0] = torch.cos(thetas)
        unitaries_coeffs[:, :, 1] = torch.sin(thetas)
        unitaries_coeffs[:, :, 2] = -torch.sin(thetas)
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

        return unitaries[:, 0].clone()

    def _construct_output_state(self, n_input_states, n_photons, state_vector):
        """Get full output state in Dirac form."""
        # TODO: move size and size_mtx to outer scope ?
        # TODO: create state_vector once at the beginning ?
        size_mtx = [self.n_individuals] + [1] * n_photons + [self.n_modes] * 2

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
        for i in range(n_photons - 1):
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

    def _extract_output_states(self, n_state_photons, n_input_states, state_vector, output_states):
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
        indices_3 = output_states[indices_3].long().reshape(-1, n_state_photons)
        c = tuple(indices_3[:, i] for i in range(n_state_photons))

        if self.n_ancilla_photons > 0:
            ancillas = self._measurements.long().reshape(self.n_individuals, self.n_success_measurements, -1)

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
                                                                 input_states.shape[1] + self.n_ancilla_photons,
                                                                 input_state_vector.to_dense())
        return self._extract_output_states(input_states.shape[1],
                                            input_states.shape[0],
                                            full_output_state_vector,
                                            output_states)

    # def crossover(self, n_offsprings):
    #     """Get random dads and moms and perform crossover."""
    #     dads = torch.randint(0, self.n_individuals, size=(n_offsprings,), device=self.device)
    #     moms = torch.randint(0, self.n_individuals, size=(n_offsprings,), device=self.device)
    #
    #     mask = torch.rand_like(self._bs_angles[moms, ...], device=self.device) < 0.5
    #     bs_angles = torch.where(mask, self._bs_angles[moms, ...], self._bs_angles[dads, ...])
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
    #     return RealPopulation(self._permutation_matrix, self._normalization_matrix, self._inverted_normalization_matrix,
    #                           bs_angles, topologies, initial_ancilla_states, measurements,
    #                           self.n_modes, self.n_ancilla_modes, self.device)
    def crossover(self, n_offsprings):
        """Get random dads and moms and perform crossover."""
        dads = torch.randint(0, self.n_individuals, size=(n_offsprings,), device=self.device)
        moms = torch.randint(0, self.n_individuals, size=(n_offsprings,), device=self.device)

        separators = torch.randint(0, self.depth, size=(n_offsprings, 1), device=self.device)
        mask = torch.zeros(n_offsprings, self.depth, dtype=torch.bool, device=self.device)
        mask.scatter_(dim=1, index=separators, value=True)
        mask = torch.cumsum(mask, dim=1).bool().view(-1, self.depth, 1)

        bs_angles = torch.where(mask.view(-1, self.depth), self._bs_angles[moms, ...], self._bs_angles[dads, ...])
        topologies = torch.where(mask, self._topologies[moms, ...], self._topologies[dads, ...])
        initial_ancilla_states = self._initial_ancilla_states[dads, ...]
        measurements = self._measurements[moms, ...]

        return RealPopulation(self._permutation_matrix, self._normalization_matrix, self._inverted_normalization_matrix,
                              bs_angles, topologies, initial_ancilla_states, measurements,
                              self.n_modes, self.n_ancilla_modes, self.device)

    def mutate(self, mutation_probability):
        """Mutate individuals."""
        mask = torch.rand_like(self._bs_angles, device=self.device) < mutation_probability
        deltas = torch.rand(size=(mask.sum().item(),), device=self.device) * tau - pi
        self._bs_angles[mask] += deltas

        mask = torch.rand_like(self._topologies, device=self.device, dtype=torch.float) < mutation_probability
        deltas = torch.randint(0, self.n_work_modes, size=(mask.sum().item(),), device=self.device)
        self._topologies[mask] += deltas

        if self.n_ancilla_modes > 0:
            mask = torch.rand_like(self._initial_ancilla_states, device=self.device, dtype=torch.float) < mutation_probability
            deltas = torch.randint(0, self.n_ancilla_modes, size=(mask.sum().item(),), device=self.device)
            self._initial_ancilla_states[mask] += deltas

            mask = torch.rand_like(self._measurements, device=self.device, dtype=torch.float) < mutation_probability
            deltas = torch.randint(0, self.n_ancilla_modes, size=(mask.sum().item(),), device=self.device)
            self._measurements[mask] += deltas

        self._normalize_data()

    def select(self, fitness, n_to_select):
        """
        Select the given number of individuals from population.
        The choice is based on the fitness.
        """
        fitness, indices = torch.topk(fitness, n_to_select)

        bs_angles = self._bs_angles[indices, ...]
        topologies = self._topologies[indices, ...]
        initial_ancilla_states = self._initial_ancilla_states[indices, ...]
        measurements = self._measurements[indices, ...]

        return RealPopulation(self._permutation_matrix, self._normalization_matrix, self._inverted_normalization_matrix,
                              bs_angles, topologies, initial_ancilla_states, measurements,
                              self.n_modes, self.n_ancilla_modes, self.device), fitness

    def __add__(self, other):
        bs_angles = torch.cat((self._bs_angles, other._bs_angles), 0)
        topologies = torch.cat((self._topologies, other._topologies), 0)
        initial_ancilla_states = torch.cat((self._initial_ancilla_states, other._initial_ancilla_states), 0)
        measurements = torch.cat((self._measurements, other._measurements), 0)

        return RealPopulation(self._permutation_matrix, self._normalization_matrix, self._inverted_normalization_matrix,
                              bs_angles, topologies, initial_ancilla_states, measurements,
                              self.n_modes, self.n_ancilla_modes, self.device)


class Population(RealPopulation):
    def __init__(self, perm_mtx, norm_mtx, inv_norm_mtx, bs_angles, ps_angles, topologies, initial_ancilla_states,
                 measurements, n_modes, n_ancilla_modes, device='cpu'):
        self._ps_angles = ps_angles
        super().__init__(perm_mtx, norm_mtx, inv_norm_mtx, bs_angles, topologies, initial_ancilla_states,
                         measurements, n_modes, n_ancilla_modes, device)

    def print(self, i):
        bs_angles = self._bs_angles[i].view(-1, 2).cpu().numpy().tolist()
        bs_angles = [f"{180. * sublist[0] / pi:.2f} {180. * sublist[1] / pi:.2f}" for sublist in bs_angles]
        ps_angles = self._ps_angles[i].view(-1).cpu().numpy().tolist()
        ps_angles = [f"{180. * angle / pi:.2f}" for angle in ps_angles]
        angles = bs_angles + ps_angles

        topology = self._topologies[i].cpu().numpy().tolist()
        topology = [f"{sublist[0]}, {sublist[1]}" for sublist in topology] + list(range(self.n_work_modes))

        elements = pd.DataFrame({'Element': ['Beam splitter'] * self.depth + ['Phase shifter'] * self.n_work_modes,
                                 'Angles': angles,
                                 'Modes': topology})

        if self.n_ancilla_photons > 0:
            modes_in = self._initial_ancilla_states[i].view(-1).cpu().numpy()
            # TODO: print all the measurements
            modes_out = self._measurements[i][0].cpu().numpy()

            ancillas = pd.DataFrame({'Mode in': modes_in,
                                     'Mode out': modes_out})
            ancillas.index.name = 'Ancilla photon'

            print(elements, ancillas, sep='\n')
        else:
            print(elements)

    def to_file(self, file_name):
        """Write data to file."""
        with open(file_name, 'w') as f:
            f.write(str(self.n_modes))
            f.write("\n")
            f.write(str(self.n_ancilla_modes))
            f.write("\n")
            f.write(json.dumps(self._bs_angles.cpu().numpy().tolist()))
            f.write("\n")
            f.write(json.dumps(self._ps_angles.cpu().numpy().tolist()))
            f.write("\n")
            f.write(json.dumps(self._topologies.cpu().numpy().tolist()))
            f.write("\n")
            f.write(json.dumps(self._initial_ancilla_states.cpu().numpy().tolist()))
            f.write("\n")
            f.write(json.dumps(self._measurements.cpu().numpy().tolist()))

    # TODO: при нескольких измерениях не допускать повторяющиеся
    def _normalize_data(self):
        """
        Bring the data to a convenient form.
        """
        self._bs_angles %= tau
        self._ps_angles %= tau

        self._topologies %= self.n_work_modes
        # self._topologies[..., 1] = torch.where(self._topologies[..., 0] == self._topologies[..., 1],
        #                                        (1 + self._topologies[..., 1]) % self.n_work_modes,
        #                                        self._topologies[..., 1])
        x = self._topologies[..., 0]
        y = self._topologies[..., 1]
        mask = x == y
        y[mask] += 1
        y[mask] %= self.n_work_modes
        self._topologies, _ = self._topologies.sort()

        if self.n_ancilla_modes > 0:
            self._initial_ancilla_states, _ = (self._initial_ancilla_states % self.n_ancilla_modes).sort()
            self._measurements, _ = (self._measurements % self.n_ancilla_modes).sort()

    def _read_scheme_unitary(self):
        """Read genome and return the unitary transformation of the scheme it represents."""
        depth = self._topologies.shape[1]

        # Indices to slice correctly
        # TODO: Move to outer scope
        indices_0 = torch.tensor([[i] * 4 * depth for i in range(self.n_individuals)], device=self.device)\
            .reshape(-1, depth, 4)
        indices_1 = torch.tensor([[i, i, i, i] for i in range(depth)] * self.n_individuals, device=self.device)\
            .reshape(-1, depth, 4)
        indices_2 = self._topologies[..., [0, 0, 1, 1]]
        indices_3 = self._topologies[..., [0, 1, 0, 1]]

        # Get angles
        thetas = self._bs_angles[..., 0]
        phis = self._bs_angles[..., 1]

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
        coeffs = self._ps_angles
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

        bs_angles = torch.where(mask, self._bs_angles[moms, ...], self._bs_angles[dads, ...])
        ps_angles = self._ps_angles[moms, ...]
        topologies = torch.where(mask, self._topologies[moms, ...], self._topologies[dads, ...])
        initial_ancilla_states = self._initial_ancilla_states[dads, ...]
        measurements = self._measurements[moms, ...]

        return Population(self._permutation_matrix, self._normalization_matrix, self._inverted_normalization_matrix,
                          bs_angles, ps_angles, topologies, initial_ancilla_states, measurements,
                          self.n_modes, self.n_ancilla_modes, self.device)

    def mutate(self, mutation_probability):
        """Mutate individuals."""
        mask = torch.rand_like(self._bs_angles, device=self.device) < mutation_probability
        deltas = torch.rand(size=(mask.sum().item(),), device=self.device) * tau - pi
        self._bs_angles[mask] += deltas

        mask = torch.rand_like(self._ps_angles, device=self.device) < mutation_probability
        deltas = torch.rand(size=(mask.sum().item(),), device=self.device) * tau - pi
        self._ps_angles[mask] += deltas

        mask = torch.rand_like(self._topologies, device=self.device, dtype=torch.float) < mutation_probability
        deltas = torch.randint(0, self.n_work_modes, size=(mask.sum().item(),), device=self.device)
        self._topologies[mask] += deltas

        if self.n_ancilla_modes > 0:
            mask = torch.rand_like(self._initial_ancilla_states, device=self.device, dtype=torch.float) < mutation_probability
            deltas = torch.randint(0, self.n_ancilla_modes, size=(mask.sum().item(),), device=self.device)
            self._initial_ancilla_states[mask] += deltas

            mask = torch.rand_like(self._measurements, device=self.device, dtype=torch.float) < mutation_probability
            deltas = torch.randint(0, self.n_ancilla_modes, size=(mask.sum().item(),), device=self.device)
            self._measurements[mask] += deltas

        self._normalize_data()

    def select(self, fitness, n_to_select):
        """
        Select the given number of individuals from population.
        The choice is based on the fitness.
        """
        fitness, indices = torch.topk(fitness, n_to_select)

        bs_angles = self._bs_angles[indices, ...]
        ps_angles = self._ps_angles[indices, ...]
        topologies = self._topologies[indices, ...]
        initial_ancilla_states = self._initial_ancilla_states[indices, ...]
        measurements = self._measurements[indices, ...]

        return Population(self._permutation_matrix, self._normalization_matrix, self._inverted_normalization_matrix,
                          bs_angles, ps_angles, topologies, initial_ancilla_states, measurements,
                          self.n_modes, self.n_ancilla_modes, self.device), fitness

    def __add__(self, other):
        bs_angles = torch.cat((self._bs_angles, other._bs_angles), 0)
        ps_angles = torch.cat((self._ps_angles, other._ps_angles), 0)
        topologies = torch.cat((self._topologies, other._topologies), 0)
        initial_ancilla_states = torch.cat((self._initial_ancilla_states, other._initial_ancilla_states), 0)
        measurements = torch.cat((self._measurements, other._measurements), 0)

        return Population(self._permutation_matrix, self._normalization_matrix, self._inverted_normalization_matrix,
                          bs_angles, ps_angles, topologies, initial_ancilla_states, measurements,
                          self.n_modes, self.n_ancilla_modes, self.device)