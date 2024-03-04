import numpy as np
import matplotlib.pyplot as plt, csv
import datetime
import data, po, model
import ast, pandas as pd
from qsee.core.ansatz import Wchain_zxz

num_assets = int(input("Enter number of assets: "))
for i in range(100):    
    print(i)
    mu, sigma, asset_list = data.get_mu_sigma('asset.csv', num_assets)
    qp = po.to_po(mu, sigma, num_assets = num_assets, q = 0.5)
    resultNumpy = model.numpy(qp)
    resultQOAO = model.qoao(qp)
    resultVQE = model.vqe(qp, Wchain_zxz(num_assets, int(num_assets/2)))

    fields = ['assets', 'list selection numpy', 'list selection qoao', 'list selection vqe', 'numpy value', 'qoao value', 'vqe value']
    row = [[str(asset_list), str(resultNumpy.x.astype(int).tolist()), str(list(resultQOAO.x.astype(int).tolist())), str(list(resultVQE.x.astype(int).tolist())), resultNumpy.fval, resultQOAO.fval, resultVQE.fval]]
    filename = f"result_{num_assets}.csv"
    df = pd.read_csv(filename)
    asset = df.assets.unique()
    if row[0][0] not in asset:
        with open(filename, 'a') as csvfile:
            # Create a csv writer object
            # Check the value of first cell in row, if it is existing, then skip writing it
            
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(row)
