from qiskit_algorithms import NumPyMinimumEigensolver, QAOA, SamplingVQE
from qiskit_algorithms.optimizers import COBYLA
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_finance.data_providers import RandomDataProvider
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import Sampler


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