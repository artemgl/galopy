import pandas as pd
import numpy as np


def print_circuit(circuit, depth, n_ancilla_photons):
    angles = circuit[:3 * depth].reshape(3, depth).t()
    angles = angles[:, [0, 2, 1]].reshape(-1)
    angles = angles.cpu().numpy() / 100.

    modes_0 = circuit[3 * depth:5 * depth:2].cpu().numpy()
    modes_0 = list(map(str, modes_0))
    modes_1 = circuit[3 * depth + 1:5 * depth:2].cpu().numpy()
    modes_1 = list(map(str, modes_1))
    modes = list(map(lambda x, y: x + ", " + y, modes_0, modes_1))
    modes = list(map(lambda x, y: [x, y, x], modes_1, modes))
    modes = [x for sublist in modes for x in sublist]

    elements = pd.DataFrame({'Element': ['Phase shifter', 'Beam splitter', 'Phase shifter'] * depth,
                             'Angle': angles,
                             'Modes': modes})

    if n_ancilla_photons > 0:
        modes_in = circuit[5 * depth:5 * depth + n_ancilla_photons].cpu().numpy()
        modes_out = circuit[5 * depth + n_ancilla_photons:5 * depth + 2 * n_ancilla_photons].cpu().numpy()

        ancillas = pd.DataFrame({'Mode in': modes_in,
                                 'Mode out': modes_out})
        ancillas.index.name = 'Ancilla photon'

        print(elements, ancillas, sep='\n')
    else:
        print(elements)


def write_circuits(path, circuits):
    circuits = pd.DataFrame(circuits)
    circuits.to_csv(path)


def read_circuits(path):
    circuits = pd.read_csv(path, index_col=0)
    return np.array(circuits.values)
