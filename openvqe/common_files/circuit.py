from qat.lang.AQASM import Program, QRoutine, RY, CNOT, RX, Z, H, RZ
from qat.core import Observable, Term, Circuit
from qat.lang.AQASM.gates import Gate
import matplotlib as mpl
from math import pi
import numpy as np
from typing import Optional, List
import warnings




def circuit_opt_simple(q, qc, exci, theta):        
    """
    A Single-fermionic-evolution circuit
    See circuit in figure 2.13 of the following reference:
    Yordanov Y.
    Quantum computational chemistry methods for early-stage quantum computers.
    PhD thesis. University of Cambridge, Cambridge, UK; 2021.
    """
    for i in range(exci[0] + 1, exci[1] - 1):
        qc.apply(CNOT, q[i], q[i + 1])

#Controlled RY gate
    qc.apply(RZ(pi/2),q[exci[0]])
    qc.apply(RY(-pi/2),q[exci[1]])
    qc.apply(RZ(-pi/2),q[exci[1]])
    qc.apply(CNOT, q[exci[0]],q[exci[1]])
    qc.apply(RY(theta),q[exci[0]])
    qc.apply(RZ(-pi/2),q[exci[1]])
    qc.apply(CNOT, q[exci[0]],q[exci[1]])
    qc.apply(RY(-theta),q[exci[0]])
    qc.apply(H,q[exci[1]])
    qc.apply(CNOT, q[exci[0]],q[exci[1]])

    for i in range(max(0, exci[1] - exci[0] - 2)):
        qc.apply(CNOT, q[exci[1] - 2 - i], q[exci[1] -1 - i])

    return qc

def circuit_opt_double(q, qc, exci, theta):            
    """
    A Double-fermionic-evolution circuit
    See circuit in figure 2.14 of the following reference:
    Yordanov Y.
    Quantum computational chemistry methods for early-stage quantum computers.
    PhD thesis. University of Cambridge, Cambridge, UK; 2021.
    """
    qc.apply(CNOT, q[exci[0]], q[exci[1]])
    qc.apply(CNOT, q[exci[2]], q[exci[3]])

    for i in range(exci[0] + 1, exci[1] - 1):
        qc.apply(CNOT, q[i], q[i + 1])

    for i in range(exci[2] + 1, exci[3] - 1):
        qc.apply(CNOT, q[i], q[i + 1])

    qc.apply(CNOT, q[exci[0]], q[exci[2]])

#Ry controlled gate
    qc.apply(RY(theta), q[exci[0]])
    qc.apply(H, q[exci[1]])
    qc.apply(CNOT, q[exci[0]], q[exci[1]])
    qc.apply(RY( - theta), q[exci[0]])
    qc.apply(H, q[exci[3]])
    qc.apply(CNOT, q[exci[0]], q[exci[3]])
    qc.apply(RY(theta), q[exci[0]])
    qc.apply(CNOT, q[exci[0]], q[exci[1]])
    qc.apply(RY( - theta), q[exci[0]])
    qc.apply(H, q[exci[2]])
    qc.apply(CNOT, q[exci[0]], q[exci[2]])
    qc.apply(RY(theta), q[exci[0]])
    qc.apply(CNOT, q[exci[0]], q[exci[1]])
    qc.apply(RY( - theta), q[exci[0]])
    qc.apply(CNOT, q[exci[0]], q[exci[3]])
    qc.apply(RY(theta), q[exci[0]])
    qc.apply(H, q[exci[3]])
    qc.apply(CNOT, q[exci[0]], q[exci[1]])
    qc.apply(RY(- 2 * theta), q[exci[0]])
    qc.apply(H, q[exci[1]])
    qc.apply(CNOT, q[exci[0]], q[exci[2]])
    qc.apply(H, q[exci[2]])

    qc.apply(CNOT, q[exci[0]], q[exci[2]])

    for i in range(max(0, exci[1] - exci[0] - 2)):
        qc.apply(CNOT, q[exci[1] - 2 - i], q[exci[1] -1 - i])

    for i in range(max(0, exci[3] - exci[2] - 2)):
        qc.apply(CNOT, q[exci[3] - 2 - i], q[exci[3] -1 - i])

    qc.apply(CNOT, q[exci[0]], q[exci[1]])
    qc.apply(CNOT, q[exci[2]], q[exci[3]])

    return qc

def efficient_fermionic_ansatz(q, qc, list_exci, list_theta):
    """
    See chapter 2 of the following reference:
    Yordanov Y.
    Quantum computational chemistry methods for early-stage quantum computers.
    PhD thesis. University of Cambridge, Cambridge, UK; 2021.
    """
    for i in range(len(list_exci)):
        if len(list_exci[i]) == 4:
            circuit_opt_double(q, qc, list_exci[i], list_theta[i])
        else :
            circuit_opt_simple(q, qc, list_exci[i], list_theta[i])
    return qc

def single_qubit_evo(q, qc, exci, theta):        
    """
    Single qubit evolution circuit
    See circuit in figure 2.10 of the following reference:
    Yordanov Y.
    Quantum computational chemistry methods for early-stage quantum computers.
    PhD thesis. University of Cambridge, Cambridge, UK; 2021.
    """

    qc.apply(RZ(pi/2),q[exci[0]])
    qc.apply(RY(-pi/2),q[exci[1]])
    qc.apply(RZ(-pi/2),q[exci[1]])
    qc.apply(CNOT, q[exci[0]],q[exci[1]])
    qc.apply(RY(theta),q[exci[0]])
    qc.apply(RZ(-pi/2),q[exci[1]])
    qc.apply(CNOT, q[exci[0]],q[exci[1]])
    qc.apply(RY(-theta),q[exci[0]])
    qc.apply(H,q[exci[1]])
    qc.apply(CNOT, q[exci[0]],q[exci[1]])

    return qc

def double_qubit_evo(q, qc, exci, theta):            
    """
    Double qubit evolution circuit
    See circuit in figure 2.14 of the following reference:
    Yordanov Y.
    Quantum computational chemistry methods for early-stage quantum computers.
    PhD thesis. University of Cambridge, Cambridge, UK; 2021.
    """

    qc.apply(CNOT, q[exci[0]], q[exci[1]])
    qc.apply(CNOT, q[exci[2]], q[exci[3]])

    qc.apply(CNOT, q[exci[0]], q[exci[2]])

    qc.apply(RY(theta), q[exci[0]])
    qc.apply(H, q[exci[1]])
    qc.apply(CNOT, q[exci[0]], q[exci[1]])
    qc.apply(RY( - theta), q[exci[0]])
    qc.apply(H, q[exci[3]])
    qc.apply(CNOT, q[exci[0]], q[exci[3]])
    qc.apply(RY(theta), q[exci[0]])
    qc.apply(CNOT, q[exci[0]], q[exci[1]])
    qc.apply(RY( - theta), q[exci[0]])
    qc.apply(H, q[exci[2]])
    qc.apply(CNOT, q[exci[0]], q[exci[2]])
    qc.apply(RY(theta), q[exci[0]])
    qc.apply(CNOT, q[exci[0]], q[exci[1]])
    qc.apply(RY( - theta), q[exci[0]])
    qc.apply(CNOT, q[exci[0]], q[exci[3]])
    qc.apply(RY(theta), q[exci[0]])
    qc.apply(H, q[exci[3]])
    qc.apply(CNOT, q[exci[0]], q[exci[1]])
    qc.apply(RY(- 2 * theta), q[exci[0]])
    qc.apply(H, q[exci[1]])
    qc.apply(CNOT, q[exci[0]], q[exci[2]])
    qc.apply(H, q[exci[2]])

    qc.apply(CNOT, q[exci[0]], q[exci[2]])

    qc.apply(CNOT, q[exci[0]], q[exci[1]])
    qc.apply(CNOT, q[exci[2]], q[exci[3]])

    return qc


def efficient_qubit_ansatz(q, qc, list_exci, list_theta):
    """
    See chapter 2 of the following reference:
    Yordanov Y.
    Quantum computational chemistry methods for early-stage quantum computers.
    PhD thesis. University of Cambridge, Cambridge, UK; 2021.
    """
    for i in range(len(list_exci)):
        if len(list_exci[i]) == 4:
            double_qubit_evo(q, qc, list_exci[i], list_theta[i])
        else :
            single_qubit_evo(q, qc, list_exci[i], list_theta[i])
    return qc

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
