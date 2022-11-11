import numpy as np
import numpy.linalg as la
from functools import reduce
from sympy import expand, sympify, latex, Float
from math import pi


# Multiply to get angle in radians from int
# TODO: move it out of here
RADIANS = pi / 18000.


def build_unitary(phi, theta, lambd, dim, mode0, mode1):
    res = np.identity(dim, dtype='complex_')
    res[mode0][mode0] = np.cos(theta)
    res[mode1][mode0] = np.sin(theta) * np.exp(1j * lambd)
    res[mode0][mode1] = -np.sin(theta) * np.exp(1j * phi)
    res[mode1][mode1] = np.cos(theta) * np.exp(1j * (phi + lambd))
    return res


#
#  Transform genome to matrix
#  genome = [step0_dev, step0_param, step0_mode1, step0_mode2, step1_dev, ..., ancilla0_in, ancilla0_out, ...]
#  dev: 0 - I, 1 - BS, 2 - PS
#
#  to_print == True - print the circuit scheme
#
# device_set = ['I', 'BS', 'PS']
def construct_circuit_matrix(genome, n_modes, depth):
    # prepare
    circuit = [np.identity(n_modes, dtype='complex_')]

    # process devices
    for i in range(depth):
        u = build_unitary(genome[i] * RADIANS, genome[2 * depth + i] * RADIANS, genome[depth + i] * RADIANS,
                          n_modes, genome[3 * depth + 2 * i], genome[3 * depth + 2 * i + 1])
        circuit.append(u)

    # matrix
    matrix = reduce(np.dot, circuit[::-1])
    # inv = la.inv(matrix)
    # return matrix, inv, ""
    return matrix


#
#  Transforms input (expression with inputs a_i) to output (expr with b_i)
#
def run_circuit(input, matrix, dim):
    for i in range(dim):
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
def use_input_ancillas(ancilla_in_ones):
    input = ""
    for i in ancilla_in_ones:
        input += "a" + str(i) + "*"
    if len(input) > 0:
        input = input[:-1]
    return input
