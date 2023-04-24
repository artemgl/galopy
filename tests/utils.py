import numpy as np
from functools import reduce
from sympy import expand, sympify


def is_valid(basic_states):
    for i in range(basic_states.shape[0]):
        for j in range(i + 1, basic_states.shape[0]):
            if np.array_equal(basic_states[i], basic_states[j]):
                return False
    return True


def gen_random_states(n_state_modes, n_state_photons, n_states):
    basic_states = np.random.randint(0, n_state_modes, size=(n_states, n_state_photons))
    while not is_valid(basic_states):
        basic_states = np.random.randint(0, n_state_modes, size=(n_states, n_state_photons))

    return basic_states


def build_bs(theta, phi, dim, mode0, mode1):
    res = np.identity(dim, dtype='complex_')
    res[mode0][mode0] = np.cos(theta)
    res[mode1][mode0] = -np.sin(theta) * np.exp(1j * phi)
    res[mode0][mode1] = np.sin(theta) * np.exp(-1j * phi)
    res[mode1][mode1] = np.cos(theta)
    return res


def build_ps(phi, dim, mode):
    res = np.identity(dim, dtype='complex_')
    res[mode][mode] = np.exp(1j * phi)
    return res


#
#  Transform genome to matrix
#  genome = [step0_dev, step0_param, step0_mode1, step0_mode2, step1_dev, ..., ancilla0_in, ancilla0_out, ...]
#  dev: 0 - I, 1 - BS, 2 - PS
#
#  to_print == True - print the circuit scheme
#
# device_set = ['I', 'BS', 'PS']
def construct_circuit_matrix(circuit, n_modes, depth):
    # prepare
    transforms = [np.identity(n_modes, dtype='complex_')]

    # process devices
    for i in range(depth):
        u = build_bs(circuit.bs_angles[i, 0], circuit.bs_angles[i, 1],
                     n_modes, circuit.topology[i, 0], circuit.topology[i, 1])
        transforms.append(u)

    for i in range(n_modes):
        u = build_ps(circuit.ps_angles[i], n_modes, i)
        transforms.append(u)

    # matrix
    matrix = reduce(np.dot, transforms[::-1])
    return matrix


#
#  Transforms input (expression with inputs a_i) to output (expr with b_i)
#
def run_circuit(input, matrix, dim):
    for i in range(dim, -1, -1):
        if "a" + str(i) in input:
            b_repr = "("
            for j in range(dim):
                b_repr += "+(" + str(matrix[j][i]) + ")*b" + str(j)
            b_repr += ")"
            input = input.replace("a" + str(i), b_repr)
    return expand(input)


#
#  Convert state from photon form to modes form
#  |000122> -> {0: 3, 1: 1, 2: 2}
#  [mode_number_0, mode_number_1, ...] -> 0: photon_count, 1: photon_count, ...
#
def photons_to_modes(index, n_modes):
    res = {i: 0 for i in range(n_modes)}
    for i in index:
        res[i] += 1
    return res


#
#  Modes expected value applied
#
def use_result(input, mode, val):
    mode_name = "b" + str(mode)
    if val == 0:
        values = {mode_name: 0}
        input = sympify(input).subs(values)
    else:
        expr = input.subs(mode_name, 0)
        input = expand(input - expr)
        input = use_result(expand(input / sympify(mode_name)), mode, val - 1)
    return input


def use_ancillas(input, photon_counts_per_modes):
    for mode in photon_counts_per_modes:
        input = use_result(input, mode, photon_counts_per_modes[mode])
    return input


def use_input(photon_counts_per_modes):
    input = ""
    for mode in photon_counts_per_modes:
        for i in range(photon_counts_per_modes[mode]):
            input += "a" + str(mode) + "*"
    if len(input) > 0:
        input = input[:-1]
    return input
