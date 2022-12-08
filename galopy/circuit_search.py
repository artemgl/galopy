import torch
from math import pi
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
        self.n_photons = input_basic_states.shape[1] + n_ancilla_photons

        self.n_success_measurements = n_success_measurements

        # Init indices for flexible slicing
        self._start_idx_rz_angles = 0
        self._start_idx_ry_angles = 2 * self.depth
        self._start_idx_modes = 3 * self.depth
        self._start_idx_ancilla_state_in = 5 * self.depth
        self._start_idx_ancilla_state_out = 5 * self.depth + self.n_ancilla_photons

    def __calculate_fidelity_and_probability(self, transforms):
        """Given transforms, get fidelity and probability for each one."""
        if self.n_success_measurements == 1:
            # Probabilities
            # TODO: изменить формулу ?
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
        transforms = population.construct_transforms(self.input_basic_states, self.output_basic_states)
        return self.__calculate_fidelity_and_probability(transforms)

    def __calculate_fitness(self, population):
        """Compute fitness for each individual in the given population."""
        fidelities, probabilities = self.__get_fidelity_and_probability(population)
        return torch.where(fidelities > 0.999, 100. * probabilities, fidelities)

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
