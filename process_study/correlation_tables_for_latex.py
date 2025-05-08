# -----------------------------------------------------
# CORRELATION TABLES
# -----------------------------------------------------
#
# This module create latex tables with QIC and standard
# errors. Matrices are used in the PhD thesis.
#
# -----------------------------------------------------


import os
module_path = os.path.abspath(os.path.join('..'))
os.chdir(module_path)


import pandas as pd
import numpy as np
import itertools

from correlation.qic import QuantileImpliedCorrelation


#%% LateX Tables - separate

# ---------------------------------
# INPUT

freq = 'M'         # D | W | M
QIC_study = pd.read_csv(f'results\studies\QIC_GLOBAL_{freq}.csv', index_col=0)
asset_price = pd.read_csv('data\GLOBAL_price.csv', parse_dates=['Date'], index_col='Date')

# ---------------------------------

tickers = asset_price.columns

alpha_lst = list(set(QIC_study.alpha)) 
risk_lst = ['VaR', 'ES']
tail_lst = ['lower', 'upper']
se = ['SE_garch', 'SE_theo']  # omitted SE_sim
perm = list(itertools.product(alpha_lst, risk_lst, tail_lst, se))

for a, r, t, se in perm:

    df = QIC_study.query(f"alpha=={a} and risk == '{r}' and tail == '{t}'")
    
    qic = df['qic']
    qic_SE = df[se]
    
    if se == 'SE_garch':
        se_name = 'GARCH bootstrapped SE'
    elif se == 'SE_theo':
        se_name = 'theoretical SE'
    
    caption = f'{a}-{r} implied correlation with {se_name} in the {t} tail'
    
    latex_table = r'\begin{table}' + '\n' + '\centering' + '\n'
    latex_table += r'\caption{' + caption + '}' + '\n'
    latex_table += r'\resizebox{\textwidth}{!}{' + '\n'
    latex_table += QuantileImpliedCorrelation.matrix_with_QIC_and_SE_latex(qic, qic_SE, tickers).to_latex(escape=False)
    latex_table += '}' + '\n' + r'\end{table}'



#%% LateX Tables - merged

# ---------------------------------
# INPUT

freq = 'M'         # D | W | M
QIC_study = pd.read_csv(f'results\studies\QIC_GLOBAL_{freq}.csv', index_col=0)
asset_price = pd.read_csv('data\GLOBAL_price.csv', parse_dates=['Date'], index_col='Date')

# Saving location
folder_path = 'results/tables/'
file_name = f'table_DAX_{str(a).replace(".", "")}_{r}_{se}_merged_tails_{freq}.tex'

# ---------------------------------

tickers = asset_price.columns


alpha_lst = list(set(QIC_study.alpha))
risk_lst = ['VaR', 'ES']
tail_lst = ['lower', 'upper']
se = ['SE_garch', 'SE_theo']  # omitted SE_sim
perm = list(itertools.product(alpha_lst, risk_lst, tail_lst, se))

for a, r, _, se in perm:
    
    if se == 'SE_garch':
        se_name = 'GARCH bootstrapped SE'
    elif se == 'SE_theo':
        se_name = 'theoretical SE'
        
    caption = f'{a}-{r} implied correlation with {se_name} - DAX - daily returns'

    df_QIC_lower = QIC_study.query(f"alpha=={a} and risk == '{r}'" + "and tail == 'lower'")
    qic_low = df_QIC_lower['qic']
    qic_low_se = df_QIC_lower[se]
    
    df_QIC_upper = QIC_study.query(f"alpha=={a} and risk == '{r}'" + "and tail == 'upper'")
    qic_upper = df_QIC_upper['qic']
    qic_upper_se = df_QIC_upper[se]

    QIC_mat_lower = QuantileImpliedCorrelation.matrix_with_QIC_and_SE_latex(qic_low, qic_low_se, tickers).values
    QIC_mat_upper = QuantileImpliedCorrelation.matrix_with_QIC_and_SE_latex(qic_upper, qic_upper_se, tickers).transpose().values
    
    QIC_mat_merged = np.zeros_like(QIC_mat_lower)
    QIC_mat_merged[np.triu_indices_from(QIC_mat_merged)] = QIC_mat_upper[np.triu_indices_from(QIC_mat_upper)]
    QIC_mat_merged[np.tril_indices_from(QIC_mat_merged)] = QIC_mat_lower[np.tril_indices_from(QIC_mat_lower)]
    
    QIC_mat_merged = pd.DataFrame(QIC_mat_merged, columns=tickers, index=tickers)
    
    latex_table = r'\begin{table}' + '\n' + '\centering' + '\n'
    latex_table += r'\caption{' + caption + '}' + '\n'
    latex_table += r'\resizebox{\textwidth}{!}{' + '\n'
    latex_table += QIC_mat_merged.to_latex(escape=False)
    latex_table += '}' + '\n' + r'\end{table}'
    
   
    with open(folder_path + file_name, 'w') as f:
        f.write(latex_table)
