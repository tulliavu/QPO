# Get the data
import pandas as pd

data = pd.read_csv("binance_data6.csv")
data
# Compute the covariance between each asset so that we can consider this values as part of our portfolio diversification constraint
from cmath import exp
import numpy as np

# Unique asset list
asset_list = data["Asset"].unique()
#expected return
exp_ret = {}
return_list = []
for asset in asset_list:
    open_price = np.array(data[data["Asset"] == asset]["Open"].astype("float"))
    close_price = np.array(data[data["Asset"] == asset]["Close"].astype("float"))
        
    # Sign will be used to indicate the value gradient direction
    returns = ((close_price - open_price)/open_price)
    exp_ret[asset] = returns.mean()
    return_list.append(returns)

# Expected return on each asset
mu = [i for i in exp_ret.values()]   
    
# Compute covariance between returns
sigma = np.cov(np.vstack(return_list))
# Here mu is the value associated with the expected average return for each asset
for i,v in zip(asset_list, mu):
    print(f"Expected average return for asset {i} is {v}")
#It is important to know what the cost is of each asset so that we can also limit the budget we would like to spend in our investment.
filter = data.groupby("Asset").agg({"Open time":max}).reset_index()
costs = data.merge(filter, how='inner').drop_duplicates()
#print(costs)
cost_list = costs[["Asset","Open"]].to_dict('records')
print(cost_list)
#We will store this information so that it can be used later.
import json

# Serializing json  
data = {"mu" : mu, "sigma": sigma.tolist(), "assets": cost_list} 
json_object = json.dumps(data, indent = 4)

with open("binance-data6.json", "w") as file:
    file.write(json_object)
# Lets load our existing data and build that optimization problem.
import json

data = None
with open("binance-data6.json", "r") as jsonfile:
    data = json.load(jsonfile)
import numpy as np

returns = data['mu']
covar = data['sigma']

assets = []
costs = []
for row in data['assets']:
    assets.append(row["Asset"])
    costs.append(float(row["Open"]))

# Half the money
budget = np.sum(costs)/0.5
import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(y=returns, x = assets)
plt.show()
#We will use CVX and its Python implementation (cvxpy) with a mixed integer optimization
# approach as our problem is restricted by the boolean values our $x$ variables can take, 
#created for convex optimization; itself isn't a solver, it is a modeling framework




import cvxpy as cp





import numpy 

# Our solution variable
x_val = cp.Variable(len(returns), boolean=True)
theta = cp.Parameter(nonneg=True)
ret = np.array(returns)@x_val
risk = cp.quad_form(x_val, covar)
e_costs = np.array(costs)@x_val

# Constraints
cons = [cp.sum(x_val) >= 0, cp.sum(e_costs) <= budget, x_val >= 0]

# Objective function
obj = cp.Minimize(- ret + theta*risk)

# Problem
prob = cp.Problem(obj, cons)
#time
import csv
import time
import pandas as pd
i = 0
with open('cvxpy6.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ['number of asset',"iteration", "time"]
    writer.writerow(field)
    while i<1000:
        i = i + 1
        start = time.process_time()
        theta.value = 0.03 # This is related to the risk penalty lagrangian
        prob.solve(solver='ECOS_BB')
        end = time.process_time()
        writer.writerow([6, i, end-start])
data1 = pd.read_csv('cvxpy6.csv')
print(data1['time'].mean())
print(data1['time'].std())
for i, val in enumerate(np.round(x_val.value,1)):
    if val == 1:
        print(f"Asset {assets[i]} was selected")
    else:
        print(f"Asset {assets[i]} was not selected")



import qiskit
import typing
def compose_circuit(qcs: typing.List[qiskit.QuantumCircuit]) -> qiskit.QuantumCircuit:
    """Combine list of paramerterized quantum circuit into one. It's very helpful!!!

    Args:
        qcs (typing.List[qiskit.QuantumCircuit]): List of quantum circuit

    Returns:
        qiskit.QuantumCircuit: composed quantum circuit
    """
    qc = qiskit.QuantumCircuit(qcs[0].num_qubits)
    i = 0
    num_params = 0
    for sub_qc in qcs:
        num_params += len(sub_qc.parameters)
    thetas = qiskit.circuit.ParameterVector('theta', num_params)
    for sub_qc in qcs:
        for instruction in sub_qc:
            if len(instruction[0].params) == 1:
                instruction[0].params[0] = thetas[i]
                i += 1
            if len(instruction[0].params) == 3:
                instruction[0].params[0] = thetas[i:i+1]
                i += 2
            qc.append(instruction[0], instruction[1])
    qc.draw()
    return qc


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


def gn(num_qubits: int, num_layers: int) -> qiskit.QuantumCircuit:
    """_summary_

    Args:
        num_qubits (int): _description_
        num_layers (int): _description_

    Returns:
        qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = compose_circuit([qc, ry_layer(num_qubits), cz_layer(num_qubits)])
    return qc


def g2gn(num_qubits: int, num_layers: int) -> qiskit.QuantumCircuit:
    """g2 + gn ansatz

    Args:
        num_qubits (int): _description_
        num_layers (int): _description_

    Returns:
        qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = compose_circuit([qc, g2(num_qubits, 1), gn(num_qubits, 1)])
    return qc


def g2gnw(num_qubits: int, num_layers: int) -> qiskit.QuantumCircuit:
    """g2 + gn + w ansatz

    Args:
        num_qubits (int): _description_
        num_layers (int): _description_

    Returns:
        qiskit.QuantumCircuit: parameterized quantum circuit
    """
    qc = qiskit.QuantumCircuit(num_qubits, num_qubits)
    for _ in range(0, num_layers):
        qc = compose_circuit([qc, g2(num_qubits, 1), gn(
            num_qubits, 1), zxz_layer(num_qubits, 1)])
    return qc

import json
import numpy as np

data = None
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

from qiskit_finance.applications.optimization import PortfolioOptimization

q = 0.5  # set risk factor
budget = len(assets) // 2  # set budget

portfolio = PortfolioOptimization(
    expected_returns=returns, covariances=covar, risk_factor=q, budget=budget
)
qp = portfolio.to_quadratic_program()
qp

#exact_eigensolver
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms import NumPyMinimumEigensolver
import time

def index_to_selection(i, num_assets):
    s = "{0:b}".format(i).rjust(num_assets)
    x = np.array([1 if s[i] == "1" else 0 for i in reversed(range(num_assets))])
    return x

def print_result(result):
    selection = result.x
    value = result.fval
    print("Optimal: selection {}, value {:.4f}".format(selection, value))

    eigenstate = result.min_eigen_solver_result.eigenstate
    
    eigenvector = eigenstate if isinstance(eigenstate, np.ndarray) else eigenstate.to_matrix()
    probabilities = np.abs(eigenvector) ** 2
    i_sorted = reversed(np.argsort(probabilities))
    print("\n----------------- Full result ---------------------")
    print("selection\tvalue\t\tprobability")
    print("---------------------------------------------------")
    for i in i_sorted:
        x = index_to_selection(i, num_assets)
        value = QuadraticProgramToQubo().convert(qp).objective.evaluate(x)
        probability = probabilities[i]
        print("%10s\t%.4f\t\t%.4f" % (x, value, probability))
exact_mes = NumPyMinimumEigensolver()
exact_eigensolver = MinimumEigenOptimizer(exact_mes)
result = exact_eigensolver.solve(qp)

#time
import csv
import time
import pandas as pd
from csv import writer
i = 0
with open('exact_eigensolver6g2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ['number of asset',"iteration", "time"]
    writer.writerow(field)
    while i<1000:
        i = i + 1
        start = time.process_time()
        result = exact_eigensolver.solve(qp)
        end = time.process_time()
        writer.writerow([6, i, end-start])
data1 = pd.read_csv('exact_eigensolver6g2.csv')
print(data1['time'].mean())
print(data1['time'].std())
print_result(result)


for i, val in enumerate(np.round(result.x,1)):
    if val == 1:
        print(f"Asset {assets[i]} was selected")
    else:
        print(f"Asset {assets[i]} was not selected")

from qiskit.circuit.library import TwoLocal
ansatz = g2(5,1)
ansatz = TwoLocal(num_assets, "ry", "cz", reps=3, entanglement="full")
ansatz.decompose().draw('mpl', fold=150)

from qiskit import Aer
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit.utils import QuantumInstance
from qiskit.utils import algorithm_globals
ansatz = g2(6,1)
algorithm_globals.random_seed = 1234
backend = Aer.get_backend("statevector_simulator")

cobyla = COBYLA()
cobyla.set_options(maxiter=500)

quantum_instance = QuantumInstance(backend=backend, seed_simulator=1234, seed_transpiler=1234)
vqe_mes = VQE(ansatz, optimizer=cobyla, quantum_instance=quantum_instance)
vqe = MinimumEigenOptimizer(vqe_mes)

#time
import csv
import time
import pandas as pd
from csv import writer
i = 0
with open('cobyla6.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ['number of asset',"iteration", "time"]
    writer.writerow(field)
    while i<1000:
        i = i + 1
        start = time.process_time()
        result = vqe.solve(qp)
        end = time.process_time()
        writer.writerow([6, i, end-start])
data1 = pd.read_csv('cobyla6.csv')
print(data1['time'].mean())
print(data1['time'].std())
print_result(result)

#ansatz = TwoLocal(num_assets, "ry", "cz", reps=6, entanglement="full")

cobyla = COBYLA()
cobyla.set_options(maxiter=500)

quantum_instance = QuantumInstance(backend=backend, seed_simulator=1234, seed_transpiler=1234)
vqe_mes = VQE(ansatz, optimizer=cobyla, quantum_instance=quantum_instance)
vqe = MinimumEigenOptimizer(vqe_mes)
result = vqe.solve(qp)

print_result(result)

#time
import csv
import time
import pandas as pd
from csv import writer
i = 0
with open('cobyla5_6reps2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ['number of asset',"iteration", "time"]
    writer.writerow(field)
    while i<1000:
        i = i + 1
        start = time.process_time()
        result = vqe.solve(qp)
        end = time.process_time()
        writer.writerow([5, i, end-start])
data1 = pd.read_csv('cobyla5_6reps2.csv')
print(data1['time'].mean())
print(data1['time'].std())
print_result(result)

best_parameters = None
for key, value in vqe_mes.__dict__.items():
    if key == "_ret":
        best_parameters = value.optimal_parameters

from qiskit.algorithms import QAOA

layers = 2
qaoa_mes = QAOA(optimizer=cobyla, reps=layers, quantum_instance=quantum_instance)
qaoa = MinimumEigenOptimizer(qaoa_mes)
result = qaoa.solve(qp)

print_result(result)


#time
import csv
import time
import pandas as pd
from csv import writer
i = 0
with open('qaoa6_2lay.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ['number of asset',"iteration", "time"]
    writer.writerow(field)
    while i<1000:
        i = i + 1
        start = time.process_time()
        result = qaoa.solve(qp)
        end = time.process_time()
        writer.writerow([6, i, end-start])
data1 = pd.read_csv('qaoa6_2lay.csv')
print(data1['time'].mean())
print(data1['time'].std())
print_result(result)

layers = 5

qaoa_mes = QAOA(optimizer=cobyla, reps=layers, quantum_instance=quantum_instance)
qaoa = MinimumEigenOptimizer(qaoa_mes)
result = qaoa.solve(qp)

print_result(result)

#time
import csv
import time
import pandas as pd
from csv import writer
i = 0
with open('qaoa5_5lay.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ['number of asset',"iteration", "time"]
    writer.writerow(field)
    while i<1000:
        i = i + 1
        start = time.process_time()
        result = qaoa.solve(qp)
        end = time.process_time()
        writer.writerow([5, i, end-start])
data1 = pd.read_csv('qaoa5_5lay.csv')
print(data1['time'].mean())
print(data1['time'].std())
print_result(result)

layers = 1

qaoa_mes = QAOA(optimizer=cobyla, reps=layers, quantum_instance=quantum_instance)
qaoa = MinimumEigenOptimizer(qaoa_mes)
result = qaoa.solve(qp)

print_result(result)

#time
import csv
import time
import pandas as pd
from csv import writer
i = 0
with open('qaoa5_1lay.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ['number of asset',"iteration", "time"]
    writer.writerow(field)
    while i<1000:
        i = i + 1
        start = time.process_time()
        result = qaoa.solve(qp)
        end = time.process_time()
        writer.writerow([5, i, end-start])
data1 = pd.read_csv('qaoa5_1lay.csv')
print(data1['time'].mean())
print(data1['time'].std())
print_result(result)

from qiskit import QuantumCircuit
ansatz = g2(5,1)
qc = QuantumCircuit(num_assets,num_assets)

# VQE Two Local ansatz
qc.compose(ansatz, inplace=True)

# Measures
for i in range(0, num_assets):
    qc.measure(i, i)

qc.draw('mpl', fold=150)

pqc = qc.bind_parameters(best_parameters.values())
pqc.draw('mpl', fold=150)

from qiskit import Aer, execute
from qiskit.visualization import plot_histogram

# Number of shots to the circuit
nshots = 10

# execute the quantum circuit
backend = Aer.get_backend('qasm_simulator') # the device to run on
result = execute(pqc, backend, shots=nshots).result()
counts  = result.get_counts(pqc)

plot_histogram(counts)

print("Solutions found:")
for cres in counts:
    print(f"Solution {cres[::-1]} with success probability {counts[cres]*100/nshots}%")
