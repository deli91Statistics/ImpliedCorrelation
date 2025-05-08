# -----------------------------------------------------
# COMPARISON QIC VS PEARSON
# -----------------------------------------------------
#
# This module loads a QIC/PEARSON study and generates
# scatterplots with qic on different alpha levels
# and a line representing the static mean pearson correlation.
#
# The idea is visualize the difference between QIC and
# the average pearson correlation
#
# -----------------------------------------------------


import os
module_path = os.path.abspath(os.path.join('..'))
os.chdir(module_path)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions.data_prep import calculate_returns
import functions.matrix_algebra as malg


#%% Generate Scatterplot of a study with one specific alpha

# ---------------------------------
# INPUT

freq = 'D'
alpha = 0.05
risk = 'ES'
tail = 'lower'

df_price = pd.read_csv('data\GLOBAL_price.csv', parse_dates=['Date'], index_col='Date')
QIC_study = pd.read_csv(f'results\studies\QIC_GLOBAL_{freq}.csv', index_col=0)

# df_price = pd.read_csv('data\DAX_19_oldest_price.csv', parse_dates=['date'], index_col='date')
# QIC_study = pd.read_csv(f'results\studies\QIC_DAX_19_log_{freq}.csv', index_col=0)


# ---------------------------------


df_returns = calculate_returns(df_price, frequency=freq, log_returns=True, demean=True)
study = QIC_study.query(f"alpha=={alpha} and risk == '{risk}' and tail=='{tail}'")
qic_sorted = study.sort_values(by='qic')

n = (df_returns.shape[1]*(df_returns.shape[1]-1))*0.5

mean_corr = np.mean(malg.vecl(df_returns.corr().values))


fig, ax = plt.subplots(figsize=(20,6))
ax.axhline(y=mean_corr, color='red', linestyle='--')
ax.scatter(range(int(n)), qic_sorted['qic'])


ax.scatter(qic_sorted['corr_pairs'], qic_sorted['lower_CI_bd'], marker="_", color='black')
ax.scatter(qic_sorted['corr_pairs'], qic_sorted['upper_CI_bd'], marker="_", color='black')
ax.set_xticklabels(qic_sorted['corr_pairs'], rotation=90)

plt.show()

#%% Generate Scatterplot of a study with all alpha levels

# ---------------------------------
# INPUT

freq = 'W'
risk = 'ES'
tail = 'lower'

# df_price = pd.read_csv('data\GLOBAL_price.csv', parse_dates=['Date'], index_col='Date')
# QIC_study = pd.read_csv(f'results\studies\QIC_GLOBAL_{freq}.csv', index_col=0)

# df_price = pd.read_csv('data\DAX_19_oldest_price.csv', parse_dates=['date'], index_col='date')
# QIC_study = pd.read_csv(f'results\studies\QIC_DAX_19_log_{freq}.csv', index_col=0)


# ---------------------------------

df_returns = calculate_returns(df_price, frequency=freq, log_returns=True, demean=True)

mean_corr = np.mean(malg.vecl(df_returns.corr().values))

alpha_lst = list(set(QIC_study.alpha))


# Create list of sorted qic values for plotting
study = [QIC_study.query(f"alpha=={a} and risk == '{risk}' and tail == '{tail}'")['qic'].sort_values() for a in alpha_lst]


fig, ax = plt.subplots(figsize=(15,8))

for plt_data, lab in zip(study, alpha_lst):
    ax.scatter(range(45), plt_data, label = lab)
    ax.plot(range(45), plt_data, linewidth = 1)
    ax.axhline(y=mean_corr, color='black', linestyle='--')
ax.legend()
ax.grid(True)
plt.show()


# %%
