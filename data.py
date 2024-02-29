import pandas as pd
import numpy as np
def get_mu_sigma(file_name, num_assets):
    data = pd.read_csv(file_name)
    # Unique asset list
    asset_list = data["Asset"].unique()
    np.random.shuffle(asset_list)
    asset_list = asset_list[:num_assets]
    #expected return
    exp_ret = {}
    return_list = []
    for asset in asset_list:
        open_price = np.array(data[data["Asset"] == asset]["Open"].astype("float"))
        close_price = np.array(data[data["Asset"] == asset]["Close"].astype("float"))
        returns = ((close_price - open_price)/open_price)
        exp_ret[asset] = returns.mean()
        return_list.append(returns)

    # Expected return on each asset
    return_list = np.array(return_list)
    mu = [i for i in exp_ret.values()]   
    # Compute covariance between returns
    sigma = np.cov((return_list))
    return mu, sigma, asset_list