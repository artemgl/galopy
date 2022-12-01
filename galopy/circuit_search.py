import torch
from math import pi, factorial
from itertools import product
from galopy.data_processing import print_circuit, write_circuits, read_circuits
from time import time

# Multiply to get angle in radians from int
# TODO: move it out of here
RADIANS = pi / 18000.


class CircuitSearch:
    def __init__(self, device: str, matrix, input_basic_states, output_basic_states=None, depth=1,
                 n_ancilla_modes=0, n_ancilla_photons=0, n_success_measurements=1):
        """
        Algorithm searching a circuit.
        Parameters:
            device: The device on which you want to store data and perform calculations (e.g. 'cuda')

            matrix: Matrix representing desired transform in the basis of basic states

            input_basic_states: Basic states on which transform is performed

            output_basic_states: Basic states which are counted as output

            depth: Number of local two-mode unitary transforms. One transform contains two phase shifters and one
            beam splitter. Must be > 0

            n_ancilla_modes: Number of modes in which ancilla photons are

            n_ancilla_photons: Number of ancilla photons

            n_success_measurements: Count of measurements that we consider as successful gate operation. Must be > 0
        """
        if n_ancilla_modes == 0 and n_ancilla_photons > 0:
            raise Exception("If number of ancilla modes is zero, number of ancilla photons must be zero as well")

        self.device = device

        self.matrix = torch.tensor(matrix, device=self.device, dtype=torch.complex64)

        input_basic_states, _ = torch.tensor(input_basic_states, device=self.device).sort()
        self.input_basic_states = input_basic_states + n_ancilla_modes
        # Number of input basic states
        self.n_input_basic_states = self.input_basic_states.shape[0]

        if not matrix.shape[1] == self.n_input_basic_states:
            raise Exception("Number of input basic states must be equal to the number of columns in transform matrix")

        if output_basic_states is None:
            self.output_basic_states = self.input_basic_states
        else:
            output_basic_states, _ = torch.tensor(output_basic_states, device=self.device).sort()
            self.output_basic_states = output_basic_states + n_ancilla_modes
        # Number of output basic states
        self.n_output_basic_states = self.output_basic_states.shape[0]

        if not matrix.shape[0] == self.n_output_basic_states:
            raise Exception("Number of output basic states must be equal to the number of rows in transform matrix")

        self.depth = depth

        self.n_state_modes = input_basic_states.max().item() + 1

        self.n_ancilla_modes = n_ancilla_modes
        # Total number of modes in scheme
        self.n_modes = self.n_state_modes + n_ancilla_modes
        # Number of modes in which unitary transform is performed
        # It's considered that all of ancilla modes always participate in this transform
        self.n_work_modes = self.n_modes

        self.n_state_photons = input_basic_states.shape[1]
        self.n_ancilla_photons = n_ancilla_photons
        # Total number of photons
        self.n_photons = self.n_state_photons + n_ancilla_photons

        self.n_success_measurements = n_success_measurements

        # Init indices for flexible slicing
        self._start_idx_rz_angles = 0
        self._start_idx_ry_angles = 2 * self.depth
        self._start_idx_modes = 3 * self.depth
        self._start_idx_ancilla_state_in = 5 * self.depth
        self._start_idx_ancilla_state_out = 5 * self.depth + self.n_ancilla_photons

        self.__precompute_extra()

    def __precompute_extra(self):
        """Compute auxiliary objects before run to optimize calculations."""
        # Precompute matrices
        self._permutation_matrix = self.__build_permutation_matrix()
        normalization_matrix, inverted_normalization_matrix =\
            self.__build_normalization_matrix(self._permutation_matrix)
        self._normalization_matrix = normalization_matrix
        self._inverted_normalization_matrix = inverted_normalization_matrix

        # Masks to get indices of modes in method __read_scheme_unitary()
        self._mask_modes_2 = torch.tensor([[self._start_idx_modes + 2 * i,
                                            self._start_idx_modes + 2 * i,
                                            self._start_idx_modes + 2 * i + 1,
                                            self._start_idx_modes + 2 * i + 1]
                                           for i in range(self.depth)], dtype=torch.long, device=self.device)
        self._mask_modes_3 = torch.tensor([[self._start_idx_modes + 2 * i,
                                            self._start_idx_modes + 2 * i + 1,
                                            self._start_idx_modes + 2 * i,
                                            self._start_idx_modes + 2 * i + 1]
                                           for i in range(self.depth)], dtype=torch.long, device=self.device)

        # Prepare mask for unitaries in method __read_scheme_unitary()
        self._mask_for_unitaries = torch.eye(self.n_modes, device=self.device, dtype=torch.bool)

    # TODO: при нескольких измерениях не допускать повторяющиеся
    def __normalize_coeffs(self, population):
        """
        Bring the data to a convenient form.

        First self.depth of parameters are angles for the first rz rotation.

        Next self.depth are angles for the second rz rotation.

        Next self.depth are angles for ry rotation.

        Next 2 * self.depth are pairs of modes on which local unitary is performed.

        Next self.n_ancilla_photons are initial state for ancilla photons.

        The last are results of measurements.
        """
        population[:, self._start_idx_rz_angles:self._start_idx_rz_angles + 2 * self.depth] %= 36000
        # Not continious in case of mutations. Consider changing 9000 to 36000
        population[:, self._start_idx_ry_angles:self._start_idx_ry_angles + self.depth] %= 9000

        population[:, self._start_idx_modes:self._start_idx_modes + 2 * self.depth] %= self.n_work_modes
        # If modes are equal, increment the second one
        x = population[:, self._start_idx_modes:self._start_idx_modes + 2 * self.depth:2]
        y = population[:, self._start_idx_modes + 1:self._start_idx_modes + 2 * self.depth:2]
        mask = y == x
        y[mask] += 1
        y[mask] %= self.n_work_modes

        if self.n_success_measurements > 0:
            if self.n_ancilla_modes > 0:
                population[:, self._start_idx_ancilla_state_in:] %= self.n_ancilla_modes

            if self.n_ancilla_photons > 0:
                modes, _ = population[:, self._start_idx_ancilla_state_in:].reshape(population.shape[0], -1,
                                                                                    self.n_ancilla_photons).sort()
                population[:, self._start_idx_ancilla_state_in:] = modes.reshape(population.shape[0], -1)

        return population

    def __gen_random_population(self, n_parents: int):
        """Create a random initial population made of n_parents."""
        res = torch.randint(0, 36000,
                            (n_parents,
                             5 * self.depth +  # 3 angles and 2 modes for each one
                             self.n_ancilla_photons * (1 +  # for initial state of ancilla photons
                                                       self.n_success_measurements)),  # results of measurements
                            device=self.device, dtype=torch.int)

        return self.__normalize_coeffs(res)

    # TODO: возможно через reshape без to_idx ?
    def __build_permutation_matrix(self):
        """
        Create matrix for output state computing.
        Multiply by it state vector to sum up all like terms.
        For example, vector (a0 * a1 + a1 * a0) will become 2 * a0 * a1
        """

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

        return torch.sparse_coo_tensor(torch.tensor(all_indices).t(), vals, device=self.device, dtype=torch.complex64)

    def __build_normalization_matrix(self, permutation_matrix):
        """
        Create matrices for transforming between two representations of state: Dirac form and operator form.
        It's considered that operator acts on the vacuum state.

            First matrix:  Dirac    -> operator ( |n> -> a^n / sqrt(n!) )

            Second matrix: operator -> Dirac    ( a^n -> sqrt(n!) * |n> )
        """
        vector = torch.ones(permutation_matrix.shape[1], 1, device=self.device, dtype=torch.complex64)
        vector = torch.sparse.mm(permutation_matrix, vector).to_sparse_coo()

        indices = vector.indices()[0].reshape(1, -1)
        indices = torch.cat((indices, indices))
        c = factorial(self.n_photons)
        norm_mtx = torch.sparse_coo_tensor(indices, (vector.values() / c).sqrt(),
                                           size=permutation_matrix.shape, device=self.device)
        inv_norm_mtx = torch.sparse_coo_tensor(indices, (c / vector.values()).sqrt(),
                                               size=permutation_matrix.shape, device=self.device)

        return norm_mtx, inv_norm_mtx

    def __read_scheme_unitary(self, population):
        """Read genome and return the unitary transformation of the scheme it represents."""
        # Indices to slice correctly
        # TODO: Move to outer scope
        indices_0 = torch.tensor([[i] * 4 * self.depth for i in range(population.shape[0])], device=self.device)\
            .reshape(-1, self.depth, 4)
        indices_1 = torch.tensor([[i, i, i, i] for i in range(self.depth)] * population.shape[0], device=self.device)\
            .reshape(-1, self.depth, 4)
        indices_2 = population[:, self._mask_modes_2]
        indices_3 = population[:, self._mask_modes_3]

        # Get angles
        phies = population[:, self._start_idx_rz_angles:self._start_idx_rz_angles + self.depth]
        thetas = population[:, self._start_idx_ry_angles:self._start_idx_ry_angles + self.depth]
        lambdas = population[:, self._start_idx_rz_angles + self.depth:self._start_idx_rz_angles + 2 * self.depth]

        # Write unitary coefficients for each triple of angles
        unitaries_coeffs = torch.zeros(population.shape[0], self.depth, 4, device=self.device, dtype=torch.complex64)
        unitaries_coeffs[:, :, 0] = torch.cos(thetas * RADIANS)
        unitaries_coeffs[:, :, 2] = torch.sin(thetas * RADIANS)
        unitaries_coeffs[:, :, 1] = -unitaries_coeffs[:, :, 2] * torch.exp(1.j * phies * RADIANS)
        unitaries_coeffs[:, :, 3] = unitaries_coeffs[:, :, 0] * torch.exp(1.j * (phies + lambdas) * RADIANS)
        unitaries_coeffs[:, :, 2] *= torch.exp(1.j * lambdas * RADIANS)

        # Create an unitary for each triple of angles
        unitaries = torch.zeros(population.shape[0], self.depth, self.n_modes, self.n_modes,
                                device=self.device, dtype=torch.complex64)
        unitaries[:, :, self._mask_for_unitaries] = 1.
        unitaries[indices_0.long(), indices_1.long(), indices_2.long(), indices_3.long()] = unitaries_coeffs

        # Multiply all the unitaries to get one for the whole scheme
        # TODO: Optimize
        for i in range(1, self.depth):
            unitaries[:, 0] = unitaries[:, i].matmul(unitaries[:, 0])

        return unitaries[:, 0].clone()

    def __construct_state(self, population):
        """Create the initial state in Dirac form."""
        # TODO: move size to outer scope ?
        size = [population.shape[0], self.n_input_basic_states] + [self.n_modes] * self.n_photons + [1]

        # TODO: outer scope ?
        sub_indices_0 = torch.tensor([[i] * self.n_input_basic_states for i in range(population.shape[0])],
                                     device=self.device).reshape(-1, 1)
        sub_indices_1 = torch.tensor([list(range(self.n_input_basic_states)) for _ in range(population.shape[0])],
                                     device=self.device).reshape(-1, 1)

        ancilla_photons =\
            population[:, self._start_idx_ancilla_state_in:self._start_idx_ancilla_state_in + self.n_ancilla_photons]

        # TODO: outer scope ?
        if self.n_ancilla_photons > 0:
            indices = torch.cat((sub_indices_0, sub_indices_1,
                                 ancilla_photons[sub_indices_0.reshape(-1)].reshape(-1, self.n_ancilla_photons),
                                 self.input_basic_states[sub_indices_1.reshape(-1)],
                                 torch.zeros_like(sub_indices_0, device=self.device)), 1).t()
        else:
            indices = torch.cat((sub_indices_0, sub_indices_1,
                                 self.input_basic_states[sub_indices_1.reshape(-1)],
                                 torch.zeros_like(sub_indices_0, device=self.device)), 1).t()

        values = torch.ones(indices.shape[1], device=self.device, dtype=torch.complex64)

        return torch.sparse_coo_tensor(indices, values, size=size, device=self.device)

    def __calculate_state(self, population):
        """Express initial state in terms of output birth operators and transform to Dirac form."""
        # TODO: move size and size_mtx to outer scope ?
        # TODO: create state_vector once at the beginning ?
        size_mtx = [population.shape[0]] + [1] * self.n_photons + [self.n_modes] * 2

        # Create state vector in Dirac form
        state_vector = self.__construct_state(population).to_dense()
        vector_shape = state_vector.shape

        # Normalize (transform to operator form)
        state_vector = state_vector.reshape(population.shape[0] * self.n_input_basic_states, -1)
        state_vector.t_()
        state_vector = torch.sparse.mm(self._normalization_matrix, state_vector)
        state_vector.t_()
        state_vector = state_vector.reshape(vector_shape)

        # Get unitary transforms
        unitaries = self.__read_scheme_unitary(population)
        unitaries = unitaries.reshape(*size_mtx)

        # Apply unitaries to all photons in state
        state_vector = unitaries.matmul(state_vector)
        # TODO: matrix reshape instead of vector transposing ?
        # TODO: Optimize? (Maybe firstly have sparse vector, then convert it to dense before some iteration)
        for i in range(self.n_photons - 1):
            state_vector.transpose_(-3 - i, -2)
            state_vector = unitaries.matmul(state_vector)
            state_vector.transpose_(-3 - i, -2)

        state_vector = state_vector.reshape(population.shape[0] * self.n_input_basic_states, -1)
        state_vector.t_()
        # TODO: Vector to sparse coo before multiplying
        # Sum up indistinguishable states with precomputed permutation matrix
        state_vector = torch.sparse.mm(self._permutation_matrix, state_vector)
        # Transform to Dirac form
        state_vector = torch.sparse.mm(self._inverted_normalization_matrix, state_vector)
        state_vector.t_()
        state_vector = state_vector.reshape(vector_shape)

        return state_vector.to_sparse_coo()

    def __construct_transforms(self, population, state_vector):
        """
        Construct transforms which are represented by circuits.
        Output matrices will be n_output_basic_states * n_input_basic_states.
        So, these matrices show how input basic states will be represented via superposition of output basic states
        after gate performing.
        """
        n_population = population.shape[0]
        n_state_photons = self.input_basic_states.shape[1]

        indices_0 = torch.tensor([[i] * self.n_input_basic_states * self.n_output_basic_states *
                                  self.n_success_measurements for i in range(n_population)],
                                 device=self.device, dtype=torch.long).reshape(-1)
        indices_1 = torch.tensor(list(range(self.n_input_basic_states))
                                 * self.n_output_basic_states * self.n_success_measurements * n_population,
                                 device=self.device, dtype=torch.long).reshape(-1)
        indices_3 = torch.tensor([[i] * self.n_input_basic_states for i in range(self.n_output_basic_states)]
                                 * n_population * self.n_success_measurements,
                                 device=self.device, dtype=torch.long).reshape(-1)

        a = (indices_0, indices_1)
        indices_3 = self.output_basic_states[indices_3].long().reshape(-1, n_state_photons)
        c = tuple(indices_3[:, i] for i in range(n_state_photons))

        if self.n_ancilla_photons > 0:
            ancillas = population[:, self._start_idx_ancilla_state_out:].long() \
                .reshape(population.shape[0], self.n_success_measurements, -1)

            indices_2 = torch.tensor([[i] * self.n_input_basic_states * self.n_output_basic_states
                                      for i in range(self.n_success_measurements)] * n_population,
                                     device=self.device, dtype=torch.long).reshape(-1)
            indices_2 = ancillas[indices_0, indices_2].reshape(-1, self.n_ancilla_photons)
            b = tuple(indices_2[:, i] for i in range(self.n_ancilla_photons))
            indices = sum((a, b, c, (0,)), ())
        else:
            indices = sum((a, c, (0,)), ())

        return state_vector.to_dense()[indices].clone().reshape(population.shape[0], self.n_success_measurements,
                                                                self.n_output_basic_states, self.n_input_basic_states)

    def __calculate_fidelity_and_probability(self, transforms):
        """Given transforms, get fidelity and probability for each one."""
        if self.n_success_measurements == 1:
            # Probabilities

            # dot = self.matrix.reshape(1, 1, self.n_output_basic_states, self.n_input_basic_states).mul(transforms)
            # dot = torch.sum(dot, 2)
            # prob_per_state = torch.abs(torch.mul(dot, dot.conj()))  # TODO: Optimize ?
            # probabilities = prob_per_state.sum(-1) / self.n_input_basic_states
            # probabilities = probabilities.reshape(-1)

            # TODO: изменить формулу
            dot = torch.abs(transforms.mul(transforms.conj()))  # TODO: Optimize ?
            prob_per_state = torch.sum(dot, 2)
            probabilities = prob_per_state.sum(-1) / self.n_input_basic_states
            probabilities = probabilities.reshape(-1)

            # Fidelities
            # Formula is taken from the article:
            # https://www.researchgate.net/publication/222547674_Fidelity_of_quantum_operations
            m = self.matrix.t().conj() \
                .reshape(1, 1, self.n_input_basic_states, self.n_output_basic_states).matmul(transforms)

            a = torch.abs(m.matmul(m.transpose(-1, -2).conj()))  # TODO: Optimize ?
            a = a.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)  # batched trace
            a = a.reshape(-1)

            b = m.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)  # batched trace
            b = torch.abs(b.mul(b.conj()))  # TODO: Optimize ?
            b = b.reshape(-1)

            fidelities = (a + b) / self.n_input_basic_states / (self.n_input_basic_states + 1)

            # The probability of gate is counted so to get real fidelity we should divide it to probability
            pure_fidelities = fidelities / probabilities
            pure_fidelities = torch.where(probabilities == 0, 0, pure_fidelities)

            return pure_fidelities, probabilities
        else:
            raise Exception("Not implemented yet! Number of success measurements should be 1 so far")

    def __get_fidelity_and_probability(self, population):
        """Given population of circuits, get fidelity and probability for each circuit."""
        state = self.__calculate_state(population)
        transforms = self.__construct_transforms(population, state)
        return self.__calculate_fidelity_and_probability(transforms)

    def __crossover(self, n_offsprings, parents):
        """Get random dads and moms from the given population and perform crossover."""
        dads = torch.randint(0, parents.shape[0], size=(n_offsprings,), device=self.device)
        dads = parents[dads, ...]

        moms = torch.randint(0, parents.shape[0], size=(n_offsprings,), device=self.device)
        moms = parents[moms, ...]

        mask = torch.rand(size=dads.shape, dtype=torch.float, device=self.device) < 0.5

        return self.__normalize_coeffs(torch.where(mask, dads, moms))

    def __mutate(self, mutation_probability, max_mutation, n_mutated, population):
        """Mutate the given number of individuals from population."""
        perm = torch.randperm(population.shape[0], device=self.device)[:n_mutated]
        sub_population = population[perm]

        mask = torch.rand(size=sub_population.shape, dtype=torch.float, device=self.device) < mutation_probability
        deltas = torch.randint(-max_mutation, max_mutation, size=(mask.sum().item(),), device=self.device)
        mutated = sub_population.clone()
        mutated[mask] += deltas
        return self.__normalize_coeffs(mutated)

    def __calculate_fitness(self, population):
        """Compute fitness for each individual in the given population."""
        fidelities, probabilities = self.__get_fidelity_and_probability(population)
        return torch.where(fidelities > 0.999, 1000. * probabilities, fidelities)

    def __select(self, population, fitness, n_to_select):
        """
        Select the given number of individuals from population.
        The choice is based on the fitness.
        """
        fitness, indices = torch.topk(fitness, n_to_select)
        return population[indices, ...], fitness

    def run(self, min_probability, n_generations, n_population, n_offsprings, n_mutated, n_elite,
            source_file=None, result_file=None):
        """
        Launch search. The algorithm stops in one of these cases:
            * After n_generations generations
            * If the circuit with fidelity > 0.999 and probability > min_probability is found

        Parameters:
            min_probability: Minimum required probability of the gate.

            n_generations: Maximum number of generations to happen.

            n_population: Number of individuals at each generation.

            n_offsprings: Number of offsprings at each generation.

            n_mutated: Number of mutated individuals at each generation. Mutation happens on random individuals,
            approximately half of genes are mutated.

            n_elite: Number of individuals with the best fitness, that are guaranteed to pass into the next
            generation.

            source_file: The file to read initial population. If is None, then random population is generated.

            result_file: The file to write the result population to. If is None, the data won't be written anywhere.
        """
        if n_elite > n_population:
            raise Exception("Number of elite can't be bigger than {threshold}.".format(threshold=n_population))

        # Save start time
        start_time = time()

        # Get initial population
        if source_file is None:
            population = self.__gen_random_population(n_population)
        else:
            circuits = read_circuits(source_file)
            n_circuits = circuits.shape[0]
            circuits = torch.tensor(circuits, device=self.device)
            if n_circuits < n_population:
                population = self.__gen_random_population(n_population - n_circuits)
                population = torch.cat((circuits, population), 0)
            else:
                population = circuits

        # Calculate fitness for the initial population
        fitness = self.__calculate_fitness(population)

        for i in range(n_generations):
            # Create new generation
            children = self.__crossover(n_offsprings, population)
            mutated = self.__mutate(0.5, 50, n_mutated, population)
            new_individuals = torch.cat((children, mutated), 0)

            # Calculate fitness for the new individuals
            new_fitness = self.__calculate_fitness(new_individuals)

            # Select individuals
            new_individuals, new_fitness = self.__select(new_individuals, new_fitness, n_population - n_elite)

            # Get best from current population
            elite_fitness, elite_indices = torch.topk(fitness, n_elite)
            elite = population[elite_indices, ...]

            # New generation
            fitness = torch.cat((elite_fitness, new_fitness), 0)
            population = torch.cat((elite, new_individuals), 0)

            print("Generation:", i + 1)
            best_fitness = fitness.max().item()
            print("Best fitness:", best_fitness)

            # If circuit with high enough fitness is found, stop
            if best_fitness >= 1000. * min_probability:
                n_generations = i + 1
                break

        # Sort result population
        fitness, indices = torch.sort(fitness, descending=True)
        population = population[indices, ...]

        # Save result population to file
        if result_file is not None:
            write_circuits(result_file, population.cpu().numpy())

        # Print result info
        print("Circuit:")
        print_circuit(population[0], self.depth, self.n_ancilla_photons)
        f, p = self.__get_fidelity_and_probability(population[0].reshape(1, -1))
        print("Fidelity: ", f[0].item())
        print("Probability: ", p[0].item())
        print(f"Processed {n_generations} generations in {time() - start_time:.2f} seconds")
