from qat.lang.AQASM import Program, QRoutine, RY, CNOT, RX, Z, H, RZ
from qat.core import Observable, Term, Circuit
from qat.lang.AQASM.gates import Gate
import matplotlib as mpl
import numpy as np
from typing import Optional, List
import warnings
​
​
def HEA_QLM(
    nqbits: int,
    #theta: List[float],
    n_cycles: int = 1,
    rotation_gates: List[Gate] = None,
    entangling_gate: Gate = CNOT,
) -> Circuit:
    
    if rotation_gates is None:
        rotation_gates = [RY]
​
    n_rotations = len(rotation_gates)
​
    prog = Program()
    reg = prog.qalloc(nqbits)
    theta = [prog.new_var(float, rf"\theta_{{{i}}}") for i in range(n_rotations * (nqbits + 2 * (nqbits - 1) * n_cycles))]
    ind_theta = 0
​
    for i in range(nqbits):
​
        for rot in rotation_gates:
​
            prog.apply(rot(theta[ind_theta]), reg[i])
            ind_theta += 1
​
    for k in range(n_cycles):
​
        for i in range(nqbits // 2):
​
            prog.apply(entangling_gate, reg[2 * i], reg[2 * i + 1])
​
            for rot in rotation_gates:
​
                prog.apply(rot(theta[ind_theta]), reg[2 * i])
                ind_theta += 1
                prog.apply(rot(theta[ind_theta]), reg[2 * i + 1])
                ind_theta += 1
​
        for i in range(nqbits // 2 - 1):
​
            prog.apply(entangling_gate, reg[2 * i + 1], reg[2 * i + 2])
​
            for rot in rotation_gates:
​
                prog.apply(rot(theta[ind_theta]), reg[2 * i + 1])
                ind_theta += 1
                prog.apply(rot(theta[ind_theta]), reg[2 * i + 2])
                ind_theta += 1
​
    return prog.to_circ()
​
def HEA_Linear(
    nqbits: int,
    #theta: List[float],
    n_cycles: int = 1,
    rotation_gates: List[Gate] = None,
    entangling_gate: Gate = CNOT,
) -> Circuit: #linear entanglement
​
    if rotation_gates is None:
        rotation_gates = [RZ]
​
    n_rotations = len(rotation_gates)
​
    prog = Program()
    reg = prog.qalloc(nqbits)
    theta = [prog.new_var(float, rf"\theta_{{{i}}}") for i in range(n_rotations * (nqbits + 2 * (nqbits - 1) * n_cycles))]
    
    
    ind_theta = 0
​
​
    
    for i in range(nqbits):
​
        for rot in rotation_gates:
​
            prog.apply(rot(theta[ind_theta]), reg[i])
            ind_theta += 1
    
    for k in range(n_cycles):
​
​
        for i in range(nqbits - 1):
            prog.apply(CNOT, reg[i], reg[i+1])
            
        for i in range(nqbits):
            for rot in rotation_gates:
                            
                prog.apply(rot(theta[ind_theta]), reg[i])
                ind_theta += 1
​
    return prog.to_circ()
​
def HEA_Full(
    nqbits: int,
    n_cycles: int = 1,
    #theta: List[float],
    rotation_gates: List[Gate] = None,
    entangling_gate: Gate = CNOT,
) -> Circuit:
​
    if rotation_gates is None:
        rotation_gates = [RZ]
​
    n_rotations = len(rotation_gates)
​
    prog = Program()
    reg = prog.qalloc(nqbits)
    #theta = [np.pi/2 for i in range(n_rotations * (nqbits + 2 * (nqbits - 1) * n_cycles))]
    theta = [prog.new_var(float, rf"\theta_{{{i}}}") for i in range(n_rotations * (nqbits + 2 * (nqbits - 1) * n_cycles))]
    
    
    ind_theta = 0
​
​
    
    for i in range(nqbits):
​
        for rot in rotation_gates:
​
            prog.apply(rot(theta[ind_theta]), reg[i])
            ind_theta += 1
    
    for k in range(n_cycles):
​
​
        for i in range(nqbits - 1):
            for j in range(i+1, nqbits):
                prog.apply(CNOT, reg[i], reg[j])
            
        for i in range(nqbits):
            for rot in rotation_gates:
                            
                prog.apply(rot(theta[ind_theta]), reg[i])
                ind_theta += 1
​
    return prog.to_circ()
​
​
def count(gate, mylist):
    """
    count function counts the number of gates in the given list
    params: it takes two parameters. first is which gate you want
    to apply like rx, ry etc. second it take the list of myqlm gates
    instruction.
    returns: it returns number of gates.
    """
    if type(gate) == type(str):
        gate = str(gate)
    if gate == gate.lower():
        gate = gate.upper()
    mylist = [str(i) for i in mylist]
    count = 0
    for i in mylist:
        if i.find("gate='{}'".format(gate)) == -1:
            pass
        else:
            count += 1
    return count
