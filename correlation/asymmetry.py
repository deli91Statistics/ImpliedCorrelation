import pandas as pd
import numpy as np
import functions.matrix_algebra as malg

from correlation.qic import QuantileImpliedCorrelation
from functions.fin_econ import filter_garch_effects, introduce_garch_effects
from functions.sampling import create_bootstrap_sample


def sample_garch_QIC_upper_vs_lower(asset_returns, benchmark_portfolios, alpha, asset_risk, portfolio_risk, iteration_size=300):

    assert alpha <= 0.5, "alpha level bounded by 0.5 for asymmetry study"
    
    garch_filtered_returns, garch_models_params = filter_garch_effects(asset_returns)                       # hard coded normal constant garch(1,1)
    garch_params_dict = {key:garch_models_params[key].params for key in garch_models_params.keys()} 
    
    QIC_lower = []
    QIC_upper = []
    
    for i in range(iteration_size):
        # IID sample
        boot_sample_iid = create_bootstrap_sample(garch_filtered_returns.values, asset_returns.shape[0])
        boot_sample_iid = dict(zip(asset_returns.columns, boot_sample_iid.T))

        # Sample with GARCH11 effects
        boot_sample_garch = {key: introduce_garch_effects(garch_params_dict[key], boot_sample_iid[key]) for key in garch_params_dict.keys()}
        boot_sample_garch = pd.DataFrame(boot_sample_garch)

        QIC_garch = QuantileImpliedCorrelation(boot_sample_garch, benchmark_portfolios)
        
        QIC_lower.append(malg.vecl(QIC_garch.qic(alpha, asset_risk, portfolio_risk, 'lower').values))
        QIC_upper.append(malg.vecl(QIC_garch.qic(1-alpha, asset_risk, portfolio_risk, 'upper').values))
        
    QIC_lower = np.array(QIC_lower)
    QIC_upper = np.array(QIC_upper)
    QIC_diff = QIC_upper - QIC_lower
        
    return QIC_lower, QIC_upper, QIC_diff


def sample_garch_QIC_vs_pearson(asset_returns, benchmark_portfolios, alpha, asset_risk, portfolio_risk, tail='lower', iteration_size=300):
    
    garch_filtered_returns, garch_models_params = filter_garch_effects(asset_returns)                       # hard coded normal constant garch(1,1)
    garch_params_dict = {key:garch_models_params[key].params for key in garch_models_params.keys()} 
    
    qic = []
    pears = []
     
    for i in range(iteration_size):
        # IID sample
        boot_sample_iid = create_bootstrap_sample(garch_filtered_returns.values, asset_returns.shape[0])
        boot_sample_iid = dict(zip(asset_returns.columns, boot_sample_iid.T))

        # Sample with GARCH11 effects
        boot_sample_garch = {key: introduce_garch_effects(garch_params_dict[key], boot_sample_iid[key]) for key in garch_params_dict.keys()}
        boot_sample_garch = pd.DataFrame(boot_sample_garch)

        QIC_garch = QuantileImpliedCorrelation(boot_sample_garch, benchmark_portfolios)
        
        qic.append(malg.vecl(QIC_garch.qic(alpha, asset_risk, portfolio_risk, tail).values))
        pears.append(malg.vecl(boot_sample_garch.corr().values))
        
    qic = np.array(qic)
    pears = np.array(pears)
    
    pearson = malg.vecl(asset_returns.corr().values)
    QIC_diff = qic - pearson
        
    return qic, pears, QIC_diff



# generalize to two generic correlation vectors, reference cor  minus experimental cor