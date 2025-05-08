import pandas as pd
import numpy as np
import itertools
from correlation.qic import QuantileImpliedCorrelation
from functions.fin_econ import weights_identification
from functions.utilities import get_correlation_pair_names
from functions.data_prep import calculate_returns


#%% ---------------------------------
# Study - DAX
# -----------------------------------

# asset_prices = pd.read_csv('data/DAX_19_oldest_price.csv', parse_dates=['date'], # index_col='date')

#freq = 'M'           # 'D', 'W', 'M'
#asset_returns = calculate_returns(asset_prices, frequency=freq, log_returns=True, demean=True)

#path = 'results/' + f'QIC_DAX_19_log_{freq}.csv'

#%% ---------------------------------
# Study - Global
# -----------------------------------
asset_prices = pd.read_csv('data/GLOBAL_price.csv', parse_dates=['Date'], index_col='Date')

freq = 'M'           # 'D', 'W', 'M'
asset_returns = calculate_returns(asset_prices, frequency=freq, log_returns=True, demean=True)

path = 'results/studies/' + f'QIC_GLOBAL_{freq}.csv'


#%% ---------------------------------
# Empirical Analysis
# -----------------------------------

p_weights = weights_identification(asset_returns.shape[1], 1)     # gradient only derived for exact identification
QIC_DAX = QuantileImpliedCorrelation(asset_returns, p_weights)

alpha_lst = [0.5, 0.05, 0.01]        # for M data, drop 0.005 and 0.001
risk_lst = ['VaR', 'ES']
tail_lst = ['lower', 'upper']
perm = list(itertools.product(alpha_lst, risk_lst, tail_lst))

corr_pair_mat = get_correlation_pair_names(asset_returns.columns)
corr_pair_names = corr_pair_mat.T[np.triu_indices(corr_pair_mat.shape[0], 1)]

QIC_DAX = QuantileImpliedCorrelation(asset_returns, p_weights)

study = []

for a, r, t in perm:
    print(a,r,t)

    QIC_DAX.qic(a, r, r, tail_area=t, SE='garch')
    qic_SE_garch = QIC_DAX.QIC_SE

    QIC_DAX.qic(a, r, r, tail_area=t, SE='theoretical')
    qic_SE_theo = QIC_DAX.QIC_SE

    QIC_DAX.qic(a, r, r, tail_area=t, SE='simulated')
    qic_SE_sim = QIC_DAX.QIC_SE
    
    conf_int = QIC_DAX.qic_confidence_interval(0.95, iteration=2000)

    qic_overview = {'corr_pairs': corr_pair_names,
                    'qic': QIC_DAX._QIC_vecl, 
                    'SE_garch': qic_SE_garch, 
                    'SE_theo': qic_SE_theo,
                    'SE_sim': qic_SE_sim,
                    'corr_class':'qic',
                    'risk':r,
                    'alpha': a, 
                    'tail':t,
                    'lower_CI_bd': conf_int['lower'],
                    'upper_CI_bd': conf_int['upper'],
                    'in_range': conf_int['in_range']}

    study.append(pd.DataFrame(qic_overview))

final = pd.concat(study)

final.to_csv(path)


#%% ========================================
# PEARSON CORRELATION RESULT FRAME
# ==========================================

import pandas as pd
import numpy as np
import functions.matrix_algebra as malg
from scipy.stats import norm

from correlation.pearson import pearson_se, sample_garch_pearson
from functions.data_prep import calculate_returns
from functions.utilities import get_correlation_pair_names

asset_price = pd.read_csv('data\DAX_19_oldest_price.csv', parse_dates=['date'], index_col='date')
# asset_price = pd.read_csv('data\GLOBAL_price.csv', parse_dates=['Date'], index_col='Date')
tickers = asset_price.columns

FR = ['D', 'W', 'M']

pearson_study = []

for freq in FR:
    asset_returns = calculate_returns(asset_price, frequency=freq, log_returns=True, demean=True)

    corr_pair_mat = get_correlation_pair_names(asset_returns.columns)
    corr_pair_names = corr_pair_mat.T[np.triu_indices(corr_pair_mat.shape[0], 1)]

    n = asset_returns.shape[0]

    p_corr = malg.vecl(asset_returns.corr().values)
    pearson_SE_garch = pd.DataFrame(sample_garch_pearson(asset_returns)).std()

    alpha = 0.95
    dist = norm(loc=0, scale=1)
    z = dist.ppf(alpha)                              # double check z-score on confidence intervals
        
    fisher_transformed_corr = np.arctanh(p_corr)
    f_lower_bound = fisher_transformed_corr - (z * pearson_SE_garch)
    f_upper_bound = fisher_transformed_corr + (z * pearson_SE_garch)
                        
    lower_bound = np.tanh(f_lower_bound)
    upper_bound = np.tanh(f_upper_bound)     
        
    pearson_overview = {'corr_pairs': corr_pair_names,
                        'cor': p_corr, 
                        'SE_garch': pearson_SE_garch.values, 
                        'SE_theo': pearson_se(p_corr, n),
                        'lower_CI_bd': lower_bound,
                        'upper_CI_bd': upper_bound,
                        'frequency': freq
                        }
    
    pearson_study.append(pd.DataFrame(pearson_overview))
    
final = pd.concat(pearson_study)

final.to_csv('results/studies/PEAR_DAX_19_log.csv')