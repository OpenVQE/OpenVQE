from qat.lang.AQASM.gates import RY, RZ, RX, CNOT
from hea import HEA_QLM, HEA_Linear, HEA_Full, count 
#After pushing the hea in circuit.py on github, we can it out by  openvqe.common_files.circuit import HEA_QLM, HEA_Lin, HEA_Full, count
​
# Example usage of HEA_QLM
nqbits_qlm = 4
n_cycles_qlm = 3
rotation_gates_qlm = [RY,RX] #  RZ  
entangling_gate_qlm = CNOT
​
circuit_qlm = HEA_QLM(nqbits_qlm, n_cycles_qlm, rotation_gates_qlm, entangling_gate_qlm)
CNOT_QLM = count("CNOT", circuit_qlm.ops)
circuit_qlm.display()
print("Number of CNOT gates in HEA_QLM circuit:", CNOT_QLM)
​
# Example usage of HEA_Linear
nqbits_linear = 4
n_cycles_linear = 3
rotation_gates_linear = [RZ]
entangling_gate_linear = CNOT
​
circuit_linear = HEA_Linear(nqbits_linear, n_cycles_linear, rotation_gates_linear, entangling_gate_linear)
CNOT_Lin = count("CNOT", circuit_linear.ops)
circuit_linear.display()
print("Number of CNOT gates in HEA_Linear circuit:", CNOT_Lin)
​
# Example usage of HEA_Full
nqbits_full = 4
n_cycles_full = 3
rotation_gates_full = [RZ]
entangling_gate_full = CNOT
​
circuit_full = HEA_Full(nqbits_full, n_cycles_full, rotation_gates_full, entangling_gate_full)
CNOT_full = count("CNOT", circuit_full.ops)
circuit_full.display()
print("Number of CNOT gates in HEA_Full circuit:", CNOT_full)
