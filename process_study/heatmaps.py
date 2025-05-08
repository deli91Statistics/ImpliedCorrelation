# -----------------------------------------------------
# HEATMAPS
# -----------------------------------------------------
#
# This module create seaborn heatmaps that visualize
# the QIC matrices. One version is for one specific case
# another version plots lower and upper tails against
# each other
#
# -----------------------------------------------------


import os
module_path = os.path.abspath(os.path.join('..'))
os.chdir(module_path)


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

import functions.matrix_algebra as malg
from functions.data_prep import calculate_returns
from correlation.qic import QuantileImpliedCorrelation


#%% Generate Pearson correlation heatmap

# ---------------------------------
# INPUT

freq = 'M'         # D | W | M
asset_price = pd.read_csv('data\GLOBAL_price.csv', parse_dates=['Date'], index_col='Date')

# ---------------------------------


asset_returns = calculate_returns(asset_price, frequency=freq, log_returns=True, demean=True)

P_matrix = asset_returns.corr()


fig, ax = plt.subplots()

ax.tick_params(axis='both', labelsize=16)
fig.set_size_inches(10, 7)
heatmap = sns.heatmap(P_matrix,
                        cmap='coolwarm', 
                        vmin=-1, 
                        vmax=1, 
                        xticklabels=True, 
                        yticklabels=True,
                            ax=ax)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=16)
plt.tight_layout()

save_path = f'results/figures/heatmap_Pearson_correlation_{freq}.png'
plt.savefig(save_path, dpi=500)


#%% Generate correlation heatmap - separate

# ---------------------------------
# INPUT

freq = 'M'         # D | W | M
asset_price = pd.read_csv('data\GLOBAL_price.csv', parse_dates=['Date'], index_col='Date')
QIC_study = pd.read_csv(f'results\studies\QIC_GLOBAL_{freq}.csv', index_col=0)

# ---------------------------------

tickers = asset_price.columns


# -------------------------------
# Heatmaps - seperated
# -------------------------------
alpha_lst = list(set(QIC_study.alpha))
risk_lst = ['VaR', 'ES']
tail_lst = ['lower', 'upper']
perm = list(itertools.product(alpha_lst, risk_lst, tail_lst))


for a, r, t in perm:
    criteria = f"alpha=={a} and risk == '{r}' and tail == '{t}'"
    df = QIC_study.query(criteria)
    
    QIC_matrix = pd.DataFrame(data=malg.antivecl(df['qic']), 
                              columns = tickers, 
                              index = tickers)
    
    QuantileImpliedCorrelation.QIC_heatmap(QIC_matrix, 
                                           asset_alpha = a, asset_risk = r, tail_area = t,
                                           title = True, annot = False)




#%% Generate correlation heatmap - merged upper/lower

# ---------------------------------
# INPUT

freq = 'M'         # D | W | M
asset_price = pd.read_csv('data\GLOBAL_price.csv', parse_dates=['Date'], index_col='Date')
QIC_study = pd.read_csv(f'results\studies\QIC_GLOBAL_{freq}.csv', index_col=0)

# ---------------------------------

tickers = asset_price.columns
   

alpha_lst = list(set(QIC_study.alpha))
risk_lst = ['VaR', 'ES']
perm = list(itertools.product(alpha_lst, risk_lst))

for a, r in perm:
    
    QIC_lower = QIC_study.query(f"alpha=={a} and risk == '{r}'" + "and tail == 'lower'")
    QIC_upper = QIC_study.query(f"alpha=={a} and risk == '{r}'" + "and tail == 'upper'")

    merged_mat = malg.antivecl_lower(QIC_lower['qic']) + malg.antivecl_lower(QIC_upper['qic']).transpose()
    np.fill_diagonal(merged_mat, 1)
    
    print(a,r)
    QIC_matrix = pd.DataFrame(data=merged_mat, columns = tickers, index = tickers)
    
    fig, ax = plt.subplots()
    ax.tick_params(axis='both', labelsize=16)
    fig.set_size_inches(10, 7)
    heatmap = sns.heatmap(QIC_matrix,
                          cmap='coolwarm', 
                          vmin=-1, 
                          vmax=1, 
                          xticklabels=True, 
                          yticklabels=True,
                          ax=ax)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    plt.tight_layout()
    
    
    folder_path = 'results/figures/'
    file_name = f'heatmap_{str(a).replace(".", "")}-{r}_QIC-merged_{freq}.png'                   
    plt.savefig(folder_path + file_name, dpi=500)
