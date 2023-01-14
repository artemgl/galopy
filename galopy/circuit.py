

class Circuit:
    def __init__(self, bs_angles, ps_angles, topology, input_ancilla_state, measurements):
        self.bs_angles = bs_angles
        self.ps_angles = ps_angles
        self.topology = topology
        self.input_ancilla_state = input_ancilla_state
        self.measurements = measurements

    def print(self):
        pass

    def to_loqc_tech(self):
        pass
