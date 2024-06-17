from galopy.gd.circuit_search import *
import numpy as np
from anytree import AnyNode, PreOrderIter, PostOrderIter


class CorrectionCircuitSearch(CircuitSearch):
    # TODO: precompute correction matrices --- лишнее, убрать
    def __init__(self, matrix, input_basic_states, output_basic_states=None, n_ancilla_modes=0, topology=None,
                 ancilla_state=None, measurements=None, device='cpu'):

        def photons_to_modes(photons_per_modes):
            modes = []
            for i in range(len(photons_per_modes)):
                n_photons = photons_per_modes[i]
                for _ in range(n_photons):
                    modes.append(i)
            return modes

        self.device = device
        self.n_state_modes = input_basic_states.max().item() + 1

        n_bs_angles = 0
        n_ps_angles = 0
        if measurements:
            self.measurements = []

            for node in PreOrderIter(measurements):
                if node.parent:
                    node.pattern = node.parent.pattern + node.pattern
                node.n_detectors = len(node.pattern)
                if not node.children:
                    self.n_ancilla_modes = node.n_detectors
                    self.n_modes = self.n_ancilla_modes + self.n_state_modes

            for node in PreOrderIter(measurements):
                if not node.children:
                    node.measurement = torch.tensor(photons_to_modes(node.pattern), device=self.device)
                    self.measurements.append(node.measurement)
                del node.pattern

                n_modes = self.n_modes - node.n_detectors
                n_bs_angles += n_modes * (n_modes - 1)
                n_ps_angles += n_modes

            self.n_measurements = len(self.measurements)
            self.measurements = torch.cat(tuple([m.view(1, -1) for m in self.measurements]))
            self.measurements_tree = measurements

        super().__init__(matrix, input_basic_states, output_basic_states, n_ancilla_modes, topology, ancilla_state,
                         self.measurements, device)

        self.correction_bs_angles = torch.nn.Linear(n_bs_angles, 1, bias=False, device=device)
        self.correction_ps_angles = torch.nn.Linear(n_ps_angles, 1, bias=False, device=device)

        # i = 0
        # j = 0
        # for node in PreOrderIter(measurements):
        #     n_modes = self.n_modes - node.n_detectors
        #     node.bs_angles = torch.nn.Linear(n_modes * (n_modes - 1), 1, bias=False, device=device)
        #     node.ps_angles = torch.nn.Linear(n_modes, 1, bias=False, device=device)
        #     # node.bs_angles = self.correction_bs_angles.weight[i:i + n_modes * (n_modes - 1), :]
        #     # node.ps_angles = self.correction_ps_angles.weight[j:j + n_modes, :]
        #
        #     i += n_modes * (n_modes - 1)
        #     j += n_modes

        # counter = 0
        # for node in PreOrderIter(self.measurements_tree):
        #     # Assign to model parameters
        #     state_dict = self.state_dict()
        #     state_dict['bs_angles_' + str(counter)] = node.bs_angles
        #     state_dict['ps_angles_' + str(counter)] = node.ps_angles
        #     self.load_state_dict(state_dict)
        #     counter += 1

        if topology is None:
            topology = tl.Parallel
        self.topology = topology

    def __construct_unitaries(self):
        i = 0
        j = 0
        for node in PostOrderIter(self.measurements_tree):
            n_modes = self.n_modes - node.n_detectors
            bs_angles = self.correction_bs_angles.weight[:, i:i + n_modes * (n_modes - 1)]
            ps_angles = self.correction_ps_angles.weight[:, j:j + n_modes]

            eye = torch.eye(node.n_detectors, device=self.device)
            u_node = self.topology(n_modes, device=self.device).\
                gen_unitary(bs_angles.view(-1, 2), ps_angles.view(-1))
            node.u = torch.block_diag(eye, u_node)

            if node.children:
                u_children = [child.u for child in node.children]
                node.u = torch.matmul(node.u, torch.cat(tuple(u_children), 1))

            i += n_modes * (n_modes - 1)
            j += n_modes

        unitaries = self.measurements_tree.u

        # for node in PreOrderIter(self.measurements):
        #     del node.u

        unitaries.t_()
        unitaries = unitaries.view(-1, self.n_modes, self.n_modes)
        unitaries.transpose_(1, 2)

        return unitaries

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
        size = [1] + [self.n_input_basic_states] + [self.n_modes] * self.n_photons + [1]

        # Create state vector in Dirac form
        state_vector = self.__construct_state().to_dense()
        vector_shape = state_vector.shape

        # Normalize (transform to operator form)
        state_vector = state_vector.reshape(self.n_input_basic_states, -1)
        state_vector.t_()
        state_vector = torch.sparse.mm(self._norm_matrix, state_vector)
        state_vector.t_()
        state_vector = state_vector.reshape(vector_shape)

        # Get unitary transforms
        unitary = self.__construct_unitaries()
        size_mtx = [self.n_measurements] + [1] * self.n_photons + [self.n_modes] * 2
        unitary = unitary.reshape(*size_mtx)

        size_expanded = [self.n_measurements] + [self.n_input_basic_states] + [self.n_modes] * self.n_photons + [1]
        state_vector = state_vector.view(*size)
        state_vector = state_vector.expand(*size_expanded)

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
        state_vector = torch.sparse.mm(self._inv_norm_matrix, state_vector)
        state_vector.t_()
        state_vector = state_vector.reshape(*size_expanded)

        # return state_vector.to_sparse_coo()
        return state_vector

    def __construct_transforms(self, state_vector):
        """
        Construct transforms which are represented by circuits.
        Output matrices will be n_output_basic_states * n_input_basic_states.
        So, these matrices show how input basic states will be represented via superposition of output basic states
        after gate performing.
        """
        state_vector = state_vector.view(self.n_measurements,
                                         self.n_input_basic_states,
                                         self.n_modes ** self.n_ancilla_photons,
                                         -1)
        state_vector.transpose_(1, 2)
        state_vector.transpose_(2, 3)

        state_vector = state_vector[:, :, self.output_basic_states_packed.view(-1).long(), :]

        if self.n_measurements > 0:
            state_vector = state_vector[:, self.measurements_packed.view(-1).long(), ...]
            state_vector = state_vector.diagonal()
            state_vector.transpose_(1, 2)
            state_vector.transpose_(0, 1)

        return state_vector

    def _calculate_fidelity_and_probability(self, transforms):
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

        # # The probability of gate is counted so to get real fidelity we should divide it to probability
        pure_fidelities = fidelities / probabilities
        pure_fidelities = torch.where(probabilities == 0, 0, pure_fidelities)

        return pure_fidelities, probabilities

    def _get_fidelity_and_probability(self):
        """Get fidelity and probability for each measurement."""
        state = self.__calculate_state()
        transforms = self.__construct_transforms(state)
        return self._calculate_fidelity_and_probability(transforms)

    def forward(self):
        return self._get_fidelity_and_probability()

    def loss(self, f, p, p_min):
        return torch.where(f > 0.99, 1. + p, f).mean()

    def run(self, min_probability, n_epochs, print_info=True):
        """
        Launch search. The algorithm stops in one of these cases:
            * After `n_epochs` epochs
            * If the circuit with fidelity > 0.999 and probability > `min_probability` is found

        Parameters:
            min_probability: Minimum required probability of the gate.

            n_epochs: Maximum number of epochs to happen.

            print_info: Whether information printing is needed.

        Returns:
            The best circuit found by the algorithm.
        """
        def pretty_metrics(f, p):
            return max(f), sum(p[f > 0.99])

        def print_progress_bar(best_f, best_p, length=40, percentage=0., reprint=False):
            f, p = pretty_metrics(best_f, best_p)
            filled = int(length * percentage)
            s = "|" + "█" * filled + " " * (length - filled) + f"| {100. * percentage:.2f}%" +\
                f"  Best fidelity: {100. * f:.2f}%" +\
                f"  Best probability: {100. * p:.2f}%"
            if reprint:
                s = "\r" + s

            print(s, end='')

        # f_history = []
        # p_history = []
        # best_circuit = copy.deepcopy(self)
        # best_f, best_p = best_circuit.forward()
        best_f, best_p = self.forward()
        # best_f = best_f.data
        # best_p = best_p.data

        if print_info:
            print_progress_bar(best_f.data.cpu().numpy(), best_p.data.cpu().numpy())

        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, maximize=True)

        for epoch_index in range(n_epochs):
            optimizer.zero_grad()

            f, p = self.forward()

            if abs(f - 1.) < 0.001:
                if p > min_probability:
                    # best_circuit = copy.deepcopy(self)
                    best_f = f
                    best_p = p
                    break
                if p > best_p:
                    # best_circuit = copy.deepcopy(self)
                    best_f = f
                    best_p = p
            else:
                if f > best_f:
                    # best_circuit = copy.deepcopy(self)
                    best_f = f
                    best_p = p

            loss_value = self.loss(f, p, min_probability)
            # f_history.append(f[0].item())
            # p_history.append(p[0].item())
            loss_value.backward()
            optimizer.step()

            if print_info:
                print_progress_bar(best_f.data.cpu().numpy(), best_p.data.cpu().numpy(),
                                   percentage=(epoch_index + 1.) / n_epochs, reprint=True)

        if print_info:
            print_progress_bar(best_f.data.cpu().numpy(), best_p.data.cpu().numpy(),
                               percentage=1., reprint=True)
            print()

        return Circuit(self.n_modes, self.n_state_modes, self.bs_angles.weight.data.view(-1, 2),
                       self.ps_angles.weight.data.view(-1), self.topology(self.n_modes).modes, self.ancilla_state,
                       self.measurements)