from math import pi
import numpy as np
import numpy.linalg as la
#from scipy.linalg import sqrtm
from functools import reduce
from sympy import expand, sympify, latex, Float
import json
from re import sub
from datetime import datetime

import sys

#
#  Linear Optics Gates
#
def BS(theta, phi, mode1, mode2, dim):
    res = np.identity(dim, dtype = 'complex_')
    res[mode1][mode1] = np.cos(theta)
    res[mode1][mode2] = -np.exp(1j*phi) * np.sin(theta)
    res[mode2][mode1] = np.exp(1j*phi) * np.sin(theta)
    res[mode2][mode2] = np.cos(theta)
    return res
def PS(phi, mode, dim):
    res = np.identity(dim, dtype = 'complex_')
    res[mode][mode] = np.exp(1j*phi) 
    return res

#
#  Transforms input (expression with inputs a_i) to output (expr with b_i)
#
def run_circuit(input, matrix, dim):
    for i in range(dim):
        if "a" + str(i+1) in input:    
            b_repr = "("
            for j in range(dim):
                b_repr += "+(" + str(matrix[i][j]) + ")*b" + str(j+1)
            b_repr += ")"
            input = input.replace("a" + str(i+1), b_repr)
    return expand(input)

#
#  Modes expected value applied
#
def use_result(input, mode, val):
    mode_name = "b" + str(mode)
    if val == 0:
        values = {mode_name: 0}
        input = sympify(input).subs(values)
    if val == 1:
        # dealing with terms having no mode-x
        values2 = {mode_name: 0}
        expr = input.subs(values2)
        input = expand(input - expr)
        # dealing with terms having more than 1 mode-x
        expr2 = expand(input / sympify(mode_name))
        expr2 = expr2.subs(values2)
        input = expr2 
    return input
def use_ancillas(input, ones, zeros):
    for i in ones:
        input = use_result(input, i, 1)
    for i in zeros:
        input = use_result(input, i, 0)
    return input
def use_input_ancillas(ancilla_in_ones):
    input = ""
    for i in ancilla_in_ones:
        input += "a" + str(i) + "*"
    if len(input) > 0:
        input = input[:-1]
    return input


#
#  Transform genome to matrix
#  genome = [step0_dev, step0_param, step0_mode1, step0_mode2, step1_dev, ..., ancilla0_in, ancilla0_out, ...]
#  dev: 0 - I, 1 - BS, 2 - PS
#
#  to_print == True - print the circuit scheme
#
device_set = ['I', 'BS', 'PS']
def ConstructCircuitMatrix(genome, modes, steps, ancillas, to_print = False):
    try:
        if not to_print:
            # prepare
            circuit = []
            circuit.append(np.identity(modes, dtype = 'complex_'))

        # process devices
        for i in range(steps):
            dev = genome[4*i] % len(device_set)
            if dev == 0:
                continue 
            if dev == 1:
                mode1 = genome[4*i + 2] % modes
                mode2 = genome[4*i + 3] % modes
                if mode1 == mode2:
                    mode2 = (mode1 + 1) % modes
                theta = genome[4*i + 1] % 90
                if to_print:
                    print("BS({mode1},{mode2}), theta={theta}".format(mode1=mode1, mode2=mode2, theta=theta))
                else:
                    circuit.append(BS(np.radians(float(theta)), 0, mode1, mode2, modes))
                continue 
            if dev == 2:
                mode1 = genome[4*i + 2] % modes
                theta = genome[4*i + 1] % 90
                if to_print:
                    print("PS({mode1}), theta={theta}".format(mode1=mode1, theta=theta))
                else:
                    circuit.append(PS(np.radians(float(theta)), mode1, modes))
                continue 

        if to_print:
            for i in range(ancillas):
                print("Acilla {num}: {a_in} - {a_out}".format(num=i, a_in=genome[4*steps + 2*i] % 2, a_out=genome[4*steps + 2*i + 1] % 2))
        else:
            # matrix
            matrix = reduce(np.dot, circuit)
            inv = la.inv(matrix)
            return matrix, inv, ""        
    except Exception as ex:
        print("ConstructCircuitMatrix: " + str(ex) )
        return None, None, "ConstructCircuitMatrix: " + str(ex)        


#
#  Fidelity & Probability part
#



def get_psi_0(input, control_modes, dest_modes):
    out = use_result(input, control_modes[1], 1)
    out = use_result(out, dest_modes[1], 1)     
    out = use_result(out, control_modes[0], 0)
    out = use_result(out, dest_modes[0], 0)     
    return complex(out)
def get_psi_1(input, control_modes, dest_modes):
    out = use_result(input, control_modes[1], 1)
    out = use_result(out, dest_modes[0], 1)     
    out = use_result(out, control_modes[0], 0)
    out = use_result(out, dest_modes[1], 0)     
    return complex(out)
def get_psi_2(input, control_modes, dest_modes):
    out = use_result(input, control_modes[0], 1)
    out = use_result(out, dest_modes[1], 1)     
    out = use_result(out, control_modes[1], 0)
    out = use_result(out, dest_modes[0], 0)     
    return complex(out)
def get_psi_3(input, control_modes, dest_modes):
    out = use_result(input, control_modes[0], 1)
    out = use_result(out, dest_modes[0], 1)     
    out = use_result(out, control_modes[1], 0)
    out = use_result(out, dest_modes[1], 0)     
    return complex(out)

def calc_psi_and_density(control_modes, dest_modes, ancilla_in_ones, ancilla_out_ones, ancilla_zeros, modes, matrix_inverse, state):
    #if not check_circuit_modes_list(control_modes, dest_modes, ancilla_in_ones, ancilla_out_ones, ancilla_zeros, modes):
    #    return None, None, "Couldn't identify conditional gate"
    input = use_input_ancillas(ancilla_in_ones)

    # prepare state
    if len(input) > 0:
        input += "*"
    if state == "00":
        input = input + "a" + str(control_modes[1]) + "*a" + str(dest_modes[1])
    if state == "01":
        input = input + "a" + str(control_modes[1]) + "*a" + str(dest_modes[0])
    if state == "10":
        input = input + "a" + str(control_modes[0]) + "*a" + str(dest_modes[1])
    if state == "11":
        input = input + "a" + str(control_modes[0]) + "*a" + str(dest_modes[0])
    input = run_circuit(input, matrix_inverse, modes)
    input = use_ancillas(input, ancilla_out_ones, ancilla_zeros)

    # vector
    psi = np.zeros(4, dtype = 'complex_')
    psi[0] = get_psi_0(input, control_modes, dest_modes)
    psi[1] = get_psi_1(input, control_modes, dest_modes)
    psi[2] = get_psi_2(input, control_modes, dest_modes)
    psi[3] = get_psi_3(input, control_modes, dest_modes)

    result = np.identity(4, dtype = 'complex_')
    for i in range(4):
        for j in range(4):
            result[i][j] = psi[i] * np.conj(psi[j])
    return result, psi, ""

def calc_psi_and_density_XY(control_modes, dest_modes, ancilla_in_ones, ancilla_out_ones, ancilla_zeros, modes, matrix_inverse, state, basis):
    temp, psi_00, error = calc_psi_and_density(control_modes, dest_modes, ancilla_in_ones, ancilla_out_ones, ancilla_zeros, modes, matrix_inverse, "00")
    if error:
        return None, None, error
    temp, psi_01, error = calc_psi_and_density(control_modes, dest_modes, ancilla_in_ones, ancilla_out_ones, ancilla_zeros, modes, matrix_inverse, "01")
    temp, psi_10, error = calc_psi_and_density(control_modes, dest_modes, ancilla_in_ones, ancilla_out_ones, ancilla_zeros, modes, matrix_inverse, "10")
    temp, psi_11, error = calc_psi_and_density(control_modes, dest_modes, ancilla_in_ones, ancilla_out_ones, ancilla_zeros, modes, matrix_inverse, "11")
    
    m1 = m2 = 1
    if state[0] == "-":
        m1 = -1
    if state[1] == "-":
        m2 = -1
    if basis == "X":
        psi = 0.5 * (psi_00 + m2*psi_01 + m1*psi_10 + m1*m2*psi_11)
    if basis == "Y":
        psi = 0.5 * (psi_00 + 1j*m2*psi_01 + 1j*m1*psi_10 - m1*m2*psi_11)

    result = np.identity(4, dtype = 'complex_')
    for i in range(4):
        for j in range(4):
            result[i][j] = psi[i] * np.conj(psi[j])
    return result, psi, ""        
    
def calc_density(control_modes, dest_modes, ancilla_in_ones, ancilla_out_ones, ancilla_zeros, modes, matrix_inverse, state, basis):
    if basis == "Z":
        res, psi, error = calc_psi_and_density(control_modes, dest_modes, ancilla_in_ones, ancilla_out_ones, ancilla_zeros, modes, matrix_inverse, state) 
    else:
        res, psi, error = calc_psi_and_density_XY(control_modes, dest_modes, ancilla_in_ones, ancilla_out_ones, ancilla_zeros, modes, matrix_inverse, state, basis) 
    return res, error


# can't use scipy in appengine  
# thanks to Tristan Nemoz for this answer:
# https://stackoverflow.com/questions/61262772/is-there-any-way-to-get-a-sqrt-of-a-matrix-in-numpy-not-element-wise-but-as-a  
def matrix_sqrt(a):
    try:
        # Computing diagonalization
        evalues, evectors = np.linalg.eig(a)
        # Ensuring square root matrix exists
        sqrt_matrix = reduce(np.dot, [evectors, np.diag(np.sqrt(evalues)), np.linalg.inv(evectors)])
        return sqrt_matrix
    except:
        # dealing with singular matrix
        return np.zeros((4, 4), dtype = 'complex_')

def calc_fidelity(rho_out, rho):
    tr_rho = np.trace(rho)
    tr_out = np.trace(rho_out)
    sqrt_rho = matrix_sqrt(rho)
    if tr_rho * tr_out == 0:
        return 0
    return abs(np.trace(matrix_sqrt(reduce(np.dot, [sqrt_rho, rho_out, sqrt_rho]))) ** 2 / (tr_rho * tr_out))

def calc_rho(gate, state):
    cgate = np.identity(4, dtype = 'complex_')
    if gate == "CZ":
        state[3] = -state[3]
    if gate == "CX":
        tmp = state[2]
        state[2] = state[3]
        state[3] = tmp
    if gate == "CY":
        tmp = 1j*state[2]
        state[2] = -1j*state[3]
        state[3] = tmp
    for i in range(4):
        for j in range(4):
            cgate[i][j] = state[i]*np.conj(state[j])
    return cgate
    
def calc_fidelity_values(rho, gate):
    result = np.zeros(13)

    #
    #   Z-base
    #
    for i in range(0, 4):
        v = np.zeros(4, dtype = "complex")
        v[i] = 1
        cz = calc_rho(gate, v)
        result[i+1] = min(1, calc_fidelity(rho[i], cz))
    #
    #   X-base
    #        
    for i in range(4, 8):
        v = np.ones(4, dtype = "complex")
        if i == 5:
            v[1] = -1
            v[3] = -1
        if i == 6:
            v[2] = -1
            v[3] = -1
        if i == 7:
            v[1] = -1
            v[2] = -1
        cz = calc_rho(gate, v)
        result[i+1] = min(1, calc_fidelity(rho[i], cz))
    #
    #   Y-base
    #        
    for i in range(8, 12):
        v = [1, 1j, 1j, -1]
        if i == 9:
            v[1] *= -1
            v[3] *= -1
        if i == 10:
            v[2] *= -1
            v[3] *= -1
        if i == 11:
            v[1] *= -1
            v[2] *= -1
        cz = calc_rho(gate, v)
        result[i+1] = min(1, calc_fidelity(rho[i], cz))

    result[0] = min(result[1:])
    return result



def get_fidelity(genome, modes = 4, steps = 10, ancillas = 0):
    states = ["00", "01", "10", "11", "++", "+-", "-+", "--", "++", "+-", "-+", "--"]
    bases = ["Z", "Z", "Z", "Z", "X", "X", "X", "X", "Y", "Y", "Y", "Y"]
    control_modes = [1, 2]
    dest_modes = [3, 4]
    ancilla_in_ones = []
    ancilla_out_ones = []
    ancilla_zeros = []
    for i in range(ancillas):
        if genome[4*steps + 2*i] % 2 == 1:
            ancilla_in_ones.append(5 + i)	
        if genome[4*steps + 2*i + 1] % 2 == 1:
            ancilla_out_ones.append(5 + i)	
        else:
            ancilla_zeros.append(5 + i)	
    error_response = [0, 0]

    matr, inv, error = ConstructCircuitMatrix(genome, modes, steps, ancillas)
    if error:
        return error_response


    Rho = [np.identity(4) for i in range(12)]
    P = np.zeros(12)
    for i in range(12):
        Rho[i], error = calc_density(control_modes, dest_modes, ancilla_in_ones, ancilla_out_ones, ancilla_zeros, modes, inv, states[i], bases[i])
        P[i] = min(1, abs(np.trace(Rho[i])))
    if error:
        return error_response
    min_prob = min(P)

    vals_cz = calc_fidelity_values(Rho, "CZ")
    min_fid = vals_cz[0]
    vals_cy = calc_fidelity_values(Rho, "CY")
    if vals_cy[0] > min_fid:
        min_fid = vals_cy[0]
    vals_cx = calc_fidelity_values(Rho, "CX")
    if vals_cx[0] > min_fid:
        min_fid = vals_cx[0]

    return [min_prob, min_fid]





import pygad

#
#  Initialization here!
#  depth - the number of linear optics gates in the circuit (gate is one of [I, BS, PS])
#  ancillas - how many ancilla modes to add
#  num_generations - number of GA generations
#  other constants - read the PyGAD guide : https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html
#
depth = 8
ancillas = 4
modes = 4 + ancillas

num_generations = 2000
num_parents_mating = 800
sol_per_pop = 5000
num_genes = modes * depth + ancillas * 2

init_range_low = 0
init_range_high = 90

parent_selection_type = "sss"
keep_parents = num_parents_mating

crossover_type = "uniform"

mutation_type = "random"
mutation_num_genes = 16
mutation_probability = 0.6

def fitness_func(solution, solution_idx):
    res = get_fidelity(solution, modes, depth, ancillas)
    if res[0] < 0.0625:
        return res[0]
    return 1000 * res[1]
def on_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Best solution:")
    ConstructCircuitMatrix(solution, modes, depth, ancillas, to_print = True)
    res = get_fidelity(solution, modes, depth, ancillas)
    print("Fitness: probability={prob}, fidelity={fid}".format(prob=res[0], fid=res[1]))

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       on_generation=on_generation,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       gene_type=int,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_num_genes=mutation_num_genes,
                       mutation_probability=mutation_probability,
                       mutation_by_replacement=True,
                       random_mutation_min_val=init_range_low,
                       random_mutation_max_val=init_range_high,
                       stop_criteria=["reach_999"],
                       parallel_processing=None)
t1 = datetime.now()
ga_instance.run()
t2 = datetime.now()
delta = t2 - t1
print(f"Time consumed: {delta.total_seconds()} seconds")


solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Best solution:")
ConstructCircuitMatrix(solution, modes, depth, ancillas, to_print = True)
res = get_fidelity(solution, modes, depth, ancillas)
print("Fitness: probability={prob}, fidelity={fid}".format(prob=res[0], fid=res[1]))
ga_instance.plot_fitness()
