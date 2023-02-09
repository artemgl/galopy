import torch
from itertools import product
from math import factorial, pi
import galopy.topology as tl
import json


class LoNet(torch.nn.Module):
    def __init__(self, matrix, input_basic_states, output_basic_states=None, n_ancilla_modes=0, ancilla_state=None,
                 measurements=None, device='cpu', topology=None):
        super().__init__()

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

        self.n_state_modes = input_basic_states.max().item() + 1
        self.n_ancilla_modes = n_ancilla_modes
        # Total number of modes in scheme
        self.n_modes = self.n_state_modes + n_ancilla_modes

        if ancilla_state is None:
            self.ancilla_state = torch.tensor([], device=self.device)
            self.n_ancilla_photons = 0
        else:
            if not ancilla_state.shape == (ancilla_state.size,):
                raise Exception("Ancilla state must be vector")
            self.ancilla_state = torch.tensor(ancilla_state, device=self.device)
            self.n_ancilla_photons = ancilla_state.size

        if measurements is None:
            self.measurements = torch.tensor([], device=self.device)
            self.n_measurements = 0
        else:
            if not measurements.shape[1] == self.n_ancilla_photons:
                raise Exception("Number of photons in measurement must be equal to number of ancilla photons")
            self.measurements = torch.tensor(measurements, device=self.device)
            self.n_measurements = measurements.shape[0]

        self.n_state_photons = input_basic_states.shape[1]
        # Total number of photons
        self.n_photons = self.n_state_photons + self.n_ancilla_photons

        if topology is None:
            self.topology = tl.Parallel(self.n_modes, device=device)
        else:
            self.topology = topology

        self.bs_angles = torch.nn.Linear(self.n_modes * (self.n_modes - 1), 1, bias=False, device=device)
        self.ps_angles = torch.nn.Linear(self.n_modes, 1, bias=False, device=device)

        self.__precompute()

    def __precompute(self):
        """Compute auxiliary objects before run to optimize calculations."""
        # Precompute matrices
        self._permutation_matrix = self.__build_permutation_matrix()
        normalization_matrix, inverted_normalization_matrix =\
            self.__build_normalization_matrix(self._permutation_matrix)
        self._normalization_matrix = normalization_matrix
        self._inverted_normalization_matrix = inverted_normalization_matrix

        w = self.n_modes ** torch.arange(self.output_basic_states.shape[1] - 1, -1, -1, device=self.device)
        self.output_basic_states_packed = self.output_basic_states.float().matmul(w.float().reshape(-1, 1)).int()

        if self.n_ancilla_photons > 0:
            w = self.n_modes ** torch.arange(self.measurements.shape[1] - 1, -1, -1, device=self.device)
            self.measurements_packed = self.measurements.float().matmul(w.float().reshape(-1, 1)).int()
        else:
            self.measurements_packed = torch.tensor([], device=self.device)

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

    def __construct_unitary(self):
        return self.topology.gen_unitary(self.bs_angles.weight.view(-1, 2), self.ps_angles.weight.view(-1))

    def __construct_state(self):
        """Create the initial state in Dirac form."""
        # TODO: move size to outer scope ?
        size = [self.n_input_basic_states] + [self.n_modes] * self.n_photons + [1]

        # TODO: outer scope ?
        sub_indices_0 = torch.tensor([0] * self.n_input_basic_states,
                                     device=self.device).reshape(-1, 1)
        sub_indices_1 = torch.tensor(list(range(self.n_input_basic_states)),
                                     device=self.device).reshape(-1, 1)

        # TODO: outer scope ?
        if self.n_ancilla_photons > 0:
            ancilla_photons = self.ancilla_state.reshape(1, -1)
            indices = torch.cat((sub_indices_1,
                                 ancilla_photons[sub_indices_0.reshape(-1)].reshape(-1, self.n_ancilla_photons),
                                 self.input_basic_states,
                                 torch.zeros_like(sub_indices_0, device=self.device)), 1).t()
        else:
            indices = torch.cat((sub_indices_1,
                                 self.input_basic_states,
                                 torch.zeros_like(sub_indices_0, device=self.device)), 1).t()

        values = torch.ones(indices.shape[1], device=self.device, dtype=torch.complex64)

        return torch.sparse_coo_tensor(indices, values, size=size, device=self.device)

    def __calculate_state(self):
        """Express initial state in terms of output birth operators and transform to Dirac form."""
        # TODO: move size and size_mtx to outer scope ?
        # TODO: create state_vector once at the beginning ?
        size_mtx = [1] * self.n_photons + [self.n_modes] * 2

        # Create state vector in Dirac form
        state_vector = self.__construct_state().to_dense()
        vector_shape = state_vector.shape

        # Normalize (transform to operator form)
        state_vector = state_vector.reshape(self.n_input_basic_states, -1)
        state_vector.t_()
        state_vector = torch.sparse.mm(self._normalization_matrix, state_vector)
        state_vector.t_()
        state_vector = state_vector.reshape(vector_shape)

        # Get unitary transforms
        unitary = self.__construct_unitary()
        unitary = unitary.reshape(*size_mtx)

        # Apply unitaries to all photons in state
        state_vector = unitary.matmul(state_vector)
        # TODO: matrix reshape instead of vector transposing ?
        # TODO: Optimize? (Maybe firstly have sparse vector, then convert it to dense before some iteration)
        for i in range(self.n_photons - 1):
            state_vector.transpose_(-3 - i, -2)
            state_vector = unitary.matmul(state_vector)
            state_vector.transpose_(-3 - i, -2)

        state_vector = state_vector.reshape(self.n_input_basic_states, -1)
        state_vector.t_()
        # TODO: Vector to sparse coo before multiplying
        # Sum up indistinguishable states with precomputed permutation matrix
        state_vector = torch.sparse.mm(self._permutation_matrix, state_vector)
        # Transform to Dirac form
        state_vector = torch.sparse.mm(self._inverted_normalization_matrix, state_vector)
        state_vector.t_()
        state_vector = state_vector.reshape(vector_shape)

        # return state_vector.to_sparse_coo()
        return state_vector

    def __construct_transforms(self, state_vector):
        """
        Construct transforms which are represented by circuits.
        Output matrices will be n_output_basic_states * n_input_basic_states.
        So, these matrices show how input basic states will be represented via superposition of output basic states
        after gate performing.
        """
        state_vector = state_vector.view(self.n_input_basic_states, self.n_modes ** self.n_ancilla_photons, -1)
        state_vector.transpose_(0, 1)
        state_vector.transpose_(1, 2)

        if self.n_measurements > 0:
            state_vector = state_vector[self.measurements_packed.view(-1).long(), ...]

        return state_vector[:, self.output_basic_states_packed.view(-1).long(), :]

    def __calculate_fidelity_and_probability(self, transforms):
        """Given transforms, get fidelity and probability for each one."""
        # Probabilities
        dot = torch.abs(transforms.mul(transforms.conj()))  # TODO: Optimize ?
        probabilities = dot.sum((1, 2)) / self.n_input_basic_states
        probabilities = probabilities.reshape(-1)

        # Fidelities
        # Formula is taken from the article:
        # https://www.researchgate.net/publication/222547674_Fidelity_of_quantum_operations
        m = self.matrix.t().conj() \
            .reshape(1, self.n_input_basic_states, self.n_output_basic_states).matmul(transforms)

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

    def __get_fidelity_and_probability(self):
        """Get fidelity and probability for each measurement."""
        state = self.__calculate_state()
        transforms = self.__construct_transforms(state)
        return self.__calculate_fidelity_and_probability(transforms)

    def forward(self):
        return self.__get_fidelity_and_probability()

    def to_loqc_tech(self, filename):
        photon_sources = []
        for i in range(self.n_modes):
            source = {
                "id": "in" + str(self.n_modes - 1 - i),
                "type": "IN",
                "theta": "undefined",
                "phi": "undefined",
                "n": "-",
                "input_type": "0",
                "x": 50,
                "y": 50 + 85 * i
            }
            photon_sources.append(source)
        for source in photon_sources[:2]:
            source["input_type"] = "2"
        for source in photon_sources[self.n_state_modes:self.n_modes]:
            source["input_type"] = "1"

        connections = []
        n_connection = 0
        frontier = [("in" + str(i), "hybrid0") for i in range(self.n_modes)]

        bs_angles = self.bs_angles.weight.data.view(-1, 2).cpu().numpy()

        counter = 0
        right_edge = 0
        beam_splitters = []
        for i, j in self.topology.modes:
            if abs(bs_angles[counter][0]) < 0.0001:
                counter += 1
                continue

            # id = "bs" + str(i) + "_" + str(j) + "_" + str(counter)
            id = "bs" + str(counter)
            beam_splitter = {
                "id": id,
                "type": "BS",
                "theta": str(180. * bs_angles[counter][0] / pi),
                "phi": str(180. * bs_angles[counter][1] / pi),
                "n": "undefined",
                "input_type": "undefined",
                "x": str(int(50 + 1500 * (right_edge + 1) / len(self.topology.modes))),
                "y": str(int(40 + 85 * (self.n_modes - 1) - 42.5 * (i + j)))
            }
            beam_splitters.append(beam_splitter)

            connection0 = {
                "id": "c" + str(n_connection),
                "type": "draw2d.Connection",
                "router": "draw2d.layout.connection.ManhattanConnectionRouter",
                "source": {"node": frontier[j][0], "port": frontier[j][1]},
                "target": {"node": id, "port": "hybrid0"}
            }
            n_connection += 1
            connection1 = {
                "id": "c" + str(n_connection),
                "type": "draw2d.Connection",
                "router": "draw2d.layout.connection.ManhattanConnectionRouter",
                "source": {"node": frontier[i][0], "port": frontier[i][1]},
                "target": {"node": id, "port": "hybrid2"}
            }
            n_connection += 1
            connections.append([connection0])
            connections.append([connection1])

            frontier[i] = id, "hybrid3"
            frontier[j] = id, "hybrid1"

            counter += 1
            right_edge += 1

        ps_angles = self.ps_angles.weight.data.view(-1).cpu().numpy()

        phase_shifters = []
        for i in range(self.n_modes):
            if abs(ps_angles[-1 - i]) < 0.0001:
                continue

            id = "ps" + str(self.n_modes - 1 - i)
            phase_shifter = {
                "id": id,
                "type": "PS",
                "theta": "undefined",
                "phi": str(180. * ps_angles[-1 - i] / pi),
                "n": "undefined",
                "input_type": "undefined",
                "x": 1650,
                "y": 50 + 85 * i
            }
            phase_shifters.append(phase_shifter)

            connection = {
                "id": "c" + str(n_connection),
                "type": "draw2d.Connection",
                "router": "draw2d.layout.connection.ManhattanConnectionRouter",
                "source": {"node": frontier[-1 - i][0], "port": frontier[-1 - i][1]},
                "target": {"node": id, "port": "hybrid0"}
            }
            n_connection += 1
            connections.append([connection])

            frontier[-1 - i] = id, "hybrid1"

        photon_detections = []
        for i in range(self.n_modes):
            id = "out" + str(self.n_modes - 1 - i)
            detection = {
                "id": id,
                "type": "OUT",
                "theta": "undefined",
                "phi": "undefined",
                "n": "-",
                "input_type": "1",
                "x": 1700,
                "y": 50 + 85 * i
            }
            photon_detections.append(detection)

            connection = {
                "id": "c" + str(n_connection),
                "type": "draw2d.Connection",
                "router": "draw2d.layout.connection.ManhattanConnectionRouter",
                "source": {"node": frontier[-1 - i][0], "port": frontier[-1 - i][1]},
                "target": {"node": id, "port": "hybrid0"}
            }
            n_connection += 1
            connections.append([connection])

        ancillas_in = [0] * self.n_ancilla_modes
        ancilla_state = self.ancilla_state.view(-1).cpu().numpy()
        for i in ancilla_state:
            ancillas_in[i] += 1
        for i in range(self.n_ancilla_modes):
            photon_sources[self.n_modes - 1 - i]["n"] = ancillas_in[i]

        ancillas_out = [0] * self.n_ancilla_modes
        measurements = self.measurements.view(-1, self.n_ancilla_photons).cpu().numpy()
        for i in measurements[0]:
            ancillas_out[i] += 1
        for i in range(self.n_ancilla_modes):
            photon_detections[self.n_modes - 1 - i]["n"] = ancillas_out[i]

        data = {
            "objects": photon_sources + beam_splitters + phase_shifters + photon_detections,
            "connections": connections
        }

        with open(filename, 'w') as f:
            f.write(json.dumps(data))

    def loss(self, f, p):
        return f.max()

    def loss2(self, f, p):
        p_min = 2. / 27.
        # p_min = 0.1062
        # p_min = 1. / 16.
        # p_min = 1. / 9.

        return ((f - p) * (1. + torch.sign(p - p_min)) + 2. * p).max()
        # return (f - torch.abs(f - p / p_min)).max()
        # return (f * p).max()
        # return torch.where(p < p_min, f, -p).max()
        # return (-p).max()
        # return f.max()
        # return p.max()
        # return torch.where(p < 1. / 9., p, f).max()
