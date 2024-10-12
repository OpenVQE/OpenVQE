import pickle
from utils import get_ham_from_dict
import matplotlib.pyplot as plt

with open("CS_hams.pickle", "rb") as handle:
    hams = pickle.load(handle)

hamiltonians = []
for n_qubits, val in hams.items():
    ham_dict = hams[n_qubits]["ham"]
    hamiltonians.append(get_ham_from_dict(ham_dict))

n_qubits = [ham.get_qubit_count() for ham in hamiltonians]
n_terms = [ham.get_term_count() for ham in hamiltonians]

plt.plot(n_qubits, n_terms, marker="o")
plt.xlabel("# qubits")
plt.ylabel("# Ham terms")
plt.xticks(n_qubits)
plt.yticks(n_terms)
plt.title("Contextual Subspace scaling")
plt.savefig("cs_scaling")
