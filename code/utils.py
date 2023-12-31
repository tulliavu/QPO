import qiskit, typing, json
import numpy as np
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit.result import QuasiDistribution
def index_to_selection(i, num_assets):
    s = "{0:b}".format(i).rjust(num_assets)
    x = np.array(
        [1 if s[i] == "1" else 0 for i in reversed(range(num_assets))])
    return x


def print_result(qp, result, num_assets):
    selection = result.x
    value = result.fval
    print("Optimal: selection {}, value {:.4f}".format(selection, value))

    eigenstate = result.min_eigen_solver_result.eigenstate
    probabilities = (
        eigenstate.binary_probabilities()
        if isinstance(eigenstate, QuasiDistribution)
        else {k: np.abs(v) ** 2 for k, v in eigenstate.to_dict().items()}
    )
    print("\n----------------- Full result ---------------------")
    print("selection\tvalue\t\tprobability")
    print("---------------------------------------------------")
    probabilities = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    for k, v in probabilities:
        x = np.array([int(i) for i in list(reversed(k))])
        value = qp.objective.evaluate(x)
        print("%10s\t%.4f\t\t%.4f" % (x, value, v))
        
def g2(num_qubits: int, num_layers: int) -> qiskit.QuantumCircuit:
    """_summary_

    Args:
        num_qubits (int): _description_
        num_layers (int): _description_

    Returns:
        qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits)
    thetas = qiskit.circuit.ParameterVector(
        'theta', 2 * num_qubits * num_layers)
    j = 0
    for _ in range(num_layers):
        for i in range(num_qubits):
            qc.ry(thetas[j], i)
            j += 1
        for i in range(0, num_qubits - 1, 2):
            qc.cz(i, i + 1)
        for i in range(num_qubits):
            qc.ry(thetas[j], i)
            j += 1
        for i in range(1, num_qubits - 1, 2):
            qc.cz(i, i + 1)
        qc.cz(0, num_qubits - 1)
    return qc


def qp_binace_data6():
    with open("binance-data6.json", "r") as jsonfile:
        data = json.load(jsonfile)
    returns = data['mu']
    covar = data['sigma']

    assets = []
    costs = []
    for row in data['assets']:
        assets.append(row["Asset"])
        costs.append(float(row["Open"]))
    num_assets = len(assets)
    q = 0.5  # set risk factor
    budget = len(assets) // 2  # set budget
    portfolio = PortfolioOptimization(
        expected_returns=returns, covariances=covar, risk_factor=q, budget=budget
    )
    qp = portfolio.to_quadratic_program()
    return qp
