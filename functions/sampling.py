import numpy as np
import pandas as pd

from functions.fin_econ import filter_garch_effects, introduce_garch_effects


def create_bootstrap_sample(data, sample_size):
    """
    Create a bootstrap sample of a specified size. First, create a random array of row indices and select from
    population sample. Every iteration, one row of the entire dataframe is selected, i.e. simultaneous draws
    See: Efron_Tibshirani(1986) for bootstrap theory
    :param data: Specified dataframe, e.g. return data
    :param sample_size: size of the dataframe
    :return:
    """
    bootstrap_index = np.random.randint(0, len(data), size=sample_size)
    bootstrap_sample = data[bootstrap_index, :]
    return bootstrap_sample


def sample_asset_portfolio_quantiles(returns, portfolios_weights, alpha, iteration_size = 200):

    garch_filtered_returns, garch_models_params = filter_garch_effects(returns)                       # hard coded normal constant garch(1,1)
    garch_params_dict = {key:garch_models_params[key].params for key in garch_models_params.keys()} 
    
    quantiles_list = []
    
    print(f'Number of bootstrap Iteration: {iteration_size}')

    for i in range(iteration_size):
        # IID sample
        boot_sample_iid = create_bootstrap_sample(garch_filtered_returns.values, returns.shape[0])
        boot_sample_iid = dict(zip(returns.columns, boot_sample_iid.T))

        # Sample with GARCH11 effects
        boot_sample_garch = {key: introduce_garch_effects(garch_params_dict[key], boot_sample_iid[key]) for key in garch_params_dict.keys()}
        boot_sample_garch = pd.DataFrame(boot_sample_garch)
        
        boot_portfolio_return = boot_sample_garch.dot(portfolios_weights.T)
        
        boot_data = np.concatenate((boot_sample_garch, boot_portfolio_return), axis=1)
            
        q = np.quantile(boot_data, alpha, method='lower', axis=0)
        
        quantiles_list.append(q)

    return np.array(quantiles_list)

