import numpy as np

def _read_module_(f, stopline, proc_fn, proc_flags):
    line = ''
    while not line.startswith (stopline):
        line = f.readline ()
        for _fn, _flag in zip (proc_fn, proc_flags):
            if line.startswith (_flag): _fn (f, line)

def read_rasscf_(f, data):
    if 'e_rasscf' not in data.keys ():
        data['e_rasscf'] = []
    flag_energy = '      Final state energy(ies):'
    def read_energy (g, line):
        e_states = []
        for i in range (3): line = g.readline ()
        while line.startswith ('::    RASSCF root number'):
            e_states.append (float (line.split ()[-1]))
            line = g.readline ()
        data['e_rasscf'].append (e_states)
    _read_module_(f, '--- Stop Module: rasscf', [read_energy,], [flag_energy,])

def read_mcpdft_(f, data):
    if 'e_mcpdft' not in data.keys ():
        data['e_mcpdft'] = []
    flag_energy = '      Total MC-PDFT energy for state'
    def read_energy (g, line):
        data['e_mcpdft'].append (float (line.split ()[-1]))
    _read_module_(f, '--- Stop Module: mcpdft', [read_energy,], [flag_energy,])

def read_alaska_(f, data):
    if 'angrad' not in data.keys ():
        data['angrad'] = []
    flag_grad = ' *              Molecular gradients               *'
    def read_grad (g, line):
        grad = []
        for i in range (8): line = g.readline ()
        while not line.startswith (' ---'):
            words = line.split ()
            grad.append ([float (w) for w in words[1:]])
            line = f.readline ()
        data['angrad'].append (np.array (grad))
    _read_module_(f, '--- Stop Module: alaska', [read_grad,], [flag_grad,])


read_module = {'&ALASKA': read_alaska_,
               '&RASSCF': read_rasscf_,
               '&MCPDFT': read_mcpdft_}
def read_molcas_logfile (fname):
    data = {}
    with open (fname, 'r') as f:
        for line in f:
            if line.startswith ('()()()'):
                line = f.readline ()
                modname = f.readline ().strip ()
                if modname in read_module.keys ():
                    read_module[modname] (f, data)
    return data

