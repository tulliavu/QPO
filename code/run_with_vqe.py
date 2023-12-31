from qiskit.visualization import plot_histogram
from qiskit import Aer, execute
from qiskit import QuantumCircuit
from qiskit.algorithms import QAOA
from qiskit.utils import algorithm_globals, QuantumInstance

from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.circuit.library import TwoLocal
from csv import writer
import pandas as pd
import csv
import json
import time
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import numpy as np
import qiskit
import typing
import utils



# time
i = 0
with open('cobyla5_6reps2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ['number of asset', "iteration", "time"]
    writer.writerow(field)
    while i < 1000:
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


layers = 2
qaoa_mes = QAOA(optimizer=cobyla, reps=layers,
                quantum_instance=quantum_instance)
qaoa = MinimumEigenOptimizer(qaoa_mes)
result = qaoa.solve(qp)

print_result(result)


# time
i = 0
with open('qaoa6_2lay.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ['number of asset', "iteration", "time"]
    writer.writerow(field)
    while i < 1000:
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

qaoa_mes = QAOA(optimizer=cobyla, reps=layers,
                quantum_instance=quantum_instance)
qaoa = MinimumEigenOptimizer(qaoa_mes)
result = qaoa.solve(qp)

print_result(result)

# time
i = 0
with open('qaoa5_5lay.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ['number of asset', "iteration", "time"]
    writer.writerow(field)
    while i < 1000:
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

qaoa_mes = QAOA(optimizer=cobyla, reps=layers,
                quantum_instance=quantum_instance)
qaoa = MinimumEigenOptimizer(qaoa_mes)
result = qaoa.solve(qp)

print_result(result)

# time
i = 0
with open('qaoa5_1lay.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ['number of asset', "iteration", "time"]
    writer.writerow(field)
    while i < 1000:
        i = i + 1
        start = time.process_time()
        result = qaoa.solve(qp)
        end = time.process_time()
        writer.writerow([5, i, end-start])
data1 = pd.read_csv('qaoa5_1lay.csv')
print(data1['time'].mean())
print(data1['time'].std())
print_result(result)

ansatz = g2(5, 1)
qc = QuantumCircuit(num_assets, num_assets)

# VQE Two Local ansatz
qc.compose(ansatz, inplace=True)

# Measures
for i in range(0, num_assets):
    qc.measure(i, i)

qc.draw('mpl', fold=150)

pqc = qc.bind_parameters(best_parameters.values())
pqc.draw('mpl', fold=150)


# Number of shots to the circuit
nshots = 10

# execute the quantum circuit
backend = Aer.get_backend('qasm_simulator')  # the device to run on
result = execute(pqc, backend, shots=nshots).result()
counts = result.get_counts(pqc)

plot_histogram(counts)

print("Solutions found:")
for cres in counts:
    print(
        f"Solution {cres[::-1]} with success probability {counts[cres]*100/nshots}%")
