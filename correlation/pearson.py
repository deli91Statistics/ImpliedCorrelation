import pandas as pd
import numpy as np
import functions.matrix_algebra as malg

from functions.fin_econ import filter_garch_effects, introduce_garch_effects
from functions.sampling import create_bootstrap_sample


def sample_garch_pearson(asset_returns, iteration_size=300):
    
    garch_filtered_returns, garch_models_params = filter_garch_effects(asset_returns)                       # hard coded normal constant garch(1,1)
    garch_params_dict = {key:garch_models_params[key].params for key in garch_models_params.keys()} 
    
    pearson_sample = []
    
    for i in range(iteration_size):
        # IID sample
        boot_sample_iid = create_bootstrap_sample(garch_filtered_returns.values, asset_returns.shape[0])
        boot_sample_iid = dict(zip(asset_returns.columns, boot_sample_iid.T))

        # Sample with GARCH11 effects
        boot_sample_garch = {key: introduce_garch_effects(garch_params_dict[key], boot_sample_iid[key]) for key in garch_params_dict.keys()}
        boot_sample_garch = pd.DataFrame(boot_sample_garch)

        pearson_sample.append(malg.vecl(boot_sample_garch.corr().values))
        
    return pearson_sample


def pearson_se(corr, n):
    """See: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    https://online.ucpress.edu/collabra/article/9/1/87615/197169/A-Brief-Note-on-the-Standard-Error-of-the-Pearson
    """
    return np.sqrt((1-corr**2)/(n-2))