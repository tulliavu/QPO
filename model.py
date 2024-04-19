from qiskit_algorithms import NumPyMinimumEigensolver, QAOA, SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_finance.data_providers import RandomDataProvider
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import Sampler
import po, numpy as np, qiskit
from qsee.core.ansatz import Wchain_zxz

def padding2(x):
    if not np.log2(len(x)).is_integer():
        next_power_of_2 = 2**np.ceil(np.log2(len(x)))
        num_zeros = int(next_power_of_2 - len(x))
        x = np.pad(x, (0, num_zeros))
    return np.array(x)
def qpo(mu, sigma):
    num_assets = len(mu)
    qp = po.to_po(mu, sigma, num_assets = num_assets, q = 0.5)
    resultVQE = numpy(qp)
    x = resultVQE.x.astype(int).tolist()
    # If len(x) is not power of 2, then add zeros to x
    x = padding2(x)
    x_k = np.linalg.norm(x)
    mu = padding2(mu)
    mu_k = np.linalg.norm(mu)
    num_qubit_assets = int(np.log2(x.shape[0]))
    qc1 = qiskit.QuantumCircuit(num_qubit_assets)
    qc1.prepare_state(x/x_k)
    qc2 = qiskit.QuantumCircuit(num_qubit_assets)
    qc2.prepare_state(mu/mu_k)

    qc = qiskit.QuantumCircuit(1 + 2 * num_qubit_assets, 1)
    qc.h(0)
    qc.compose(qc1, qubits=[*range(1, 1 + num_qubit_assets)], inplace=True)
    qc.compose(qc2, qubits=[*range(1 + num_qubit_assets, 1 + num_qubit_assets * 2)], inplace=True)
    for i in range(1, num_qubit_assets + 1):
        qc.cswap(0, i, i + num_qubit_assets)
    qc.h(0)
    qc.measure(0, 0)
    sampler = Sampler()
    result = sampler.run(qc, shots = 10000).result().quasi_dists[0]
    return np.sqrt(result.get(0, 0) - result.get(1, 0))*x_k*mu_k, abs(np.inner(x, mu))

def numpy(qp):
    exact_mes = NumPyMinimumEigensolver()
    exact_eigensolver = MinimumEigenOptimizer(exact_mes)

    result = exact_eigensolver.solve(qp)
    return result

def vqe(qp, ansatz):
    cobyla = COBYLA()
    cobyla.set_options(maxiter=1000)
    svqe_mes = SamplingVQE(sampler=Sampler(), ansatz=ansatz, optimizer=cobyla)
    svqe = MinimumEigenOptimizer(svqe_mes)
    result = svqe.solve(qp)
    return result

def qoao(qp):
    cobyla = COBYLA()
    cobyla.set_options(maxiter=1000)
    qaoa_mes = QAOA(sampler=Sampler(), optimizer=cobyla, reps=3)
    qaoa = MinimumEigenOptimizer(qaoa_mes)
    result = qaoa.solve(qp)
    return result