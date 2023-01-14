import torch
from math import tau, pi
import json
from galopy.population_type import Population, RealPopulation


def random(perm_mtx, norm_mtx, inv_norm_mtx, n_individuals=1, depth=1, n_modes=2, n_ancilla_modes=0,
           n_ancilla_photons=0, n_success_measurements=0, device='cpu', ptype='universal'):
    if ptype == 'universal':
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

        return Population(perm_mtx, norm_mtx, inv_norm_mtx,
                          bs_angles, ps_angles, topologies, initial_ancilla_states, measurements,
                          n_modes, n_ancilla_modes, device)
    elif ptype == 'real':
        bs_angles = torch.rand(n_individuals, depth, device=device) * tau

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

        return RealPopulation(perm_mtx, norm_mtx, inv_norm_mtx,
                              bs_angles, topologies, initial_ancilla_states, measurements,
                              n_modes, n_ancilla_modes, device)
    else:
        raise Exception("Unsupported population type.")


def from_file(file_name, perm_mtx, norm_mtx, inv_norm_mtx, device='cpu', ptype='universal'):
    if ptype == 'universal':
        with open(file_name, 'r') as f:
            n_modes = int(f.readline())
            n_ancilla_modes = int(f.readline())
            bs_angles = torch.tensor(json.loads(f.readline()), device=device)
            ps_angles = torch.tensor(json.loads(f.readline()), device=device)
            topologies = torch.tensor(json.loads(f.readline()), device=device, dtype=torch.int8)
            initial_ancilla_states = torch.tensor(json.loads(f.readline()), device=device, dtype=torch.int8)
            measurements = torch.tensor(json.loads(f.readline()), device=device, dtype=torch.int8)

        return Population(perm_mtx, norm_mtx, inv_norm_mtx,
                          bs_angles, ps_angles, topologies, initial_ancilla_states, measurements,
                          n_modes, n_ancilla_modes, device)
    elif ptype == 'real':
        with open(file_name, 'r') as f:
            n_modes = int(f.readline())
            n_ancilla_modes = int(f.readline())
            bs_angles = torch.tensor(json.loads(f.readline()), device=device)
            topologies = torch.tensor(json.loads(f.readline()), device=device, dtype=torch.int8)
            initial_ancilla_states = torch.tensor(json.loads(f.readline()), device=device, dtype=torch.int8)
            measurements = torch.tensor(json.loads(f.readline()), device=device, dtype=torch.int8)

        return RealPopulation(perm_mtx, norm_mtx, inv_norm_mtx,
                              bs_angles, topologies, initial_ancilla_states, measurements,
                              n_modes, n_ancilla_modes, device)
    else:
        raise Exception("Unsupported population type.")
