import numpy as np
from qiskit_finance.applications.optimization import PortfolioOptimization
def to_po(mu, sigma, **kwargs):
    num_assets = kwargs['num_assets']  # set number of assets
    q =  kwargs['q'] # set risk factor
    budget = num_assets // 2  # set budget
    penalty = num_assets  # set parameter to scale the budget penalty term

    portfolio = PortfolioOptimization(
        expected_returns=mu, covariances=sigma, risk_factor=q, budget=budget
    )
    qp = portfolio.to_quadratic_program()
    return qp

def hamming(vector1, vector2):
    """Calculate the Hamming distance between two binary vectors"""
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must be of the same length")
    
    distance = 0
    for bit1, bit2 in zip(vector1, vector2):
        if bit1 != bit2:
            distance += 1
    
    return distance/len(vector1)