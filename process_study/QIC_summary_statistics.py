# -----------------------------------------------------
# SUMMARY STATISTICS
# -----------------------------------------------------
#
# This module loads a QIC/PEARSON study and computes the summary 
# statistics of the corresponding correlation pairs for
# the specified set of parameters. The results were
# copied into an external latex table.
#
# -----------------------------------------------------


import os
module_path = os.path.abspath(os.path.join('..'))
os.chdir(module_path)

import pandas as pd


#%% Summary Statistics - QIC

# ---------------------------------
# INPUT

QIC_study = pd.read_csv('results\studies\QIC_GLOBAL_D.csv', index_col=0)

# ---------------------------------


alpha_lst = list(set(QIC_study.alpha))

# Results from lower tail
for a in alpha_lst:
    df = QIC_study.query(f"alpha=={a}")
    
    df_lower = df.query("tail == 'lower'")
    df_lower_VaR = df_lower.query("risk == 'VaR'")
    df_lower_ES = df_lower.query("risk == 'ES'")
    
    print(a)
    print('lower')
    print(f'{round(df_lower_VaR.qic.mean(),4)} & {round(df_lower_ES.qic.mean(),4)} & ~ & {round(df_lower_VaR.SE_theo.mean(), 4)} & {round(df_lower_ES.SE_theo.mean(),4)} & ~ & {round(df_lower_VaR.SE_garch.mean(), 4)} & {round(df_lower_ES.SE_garch.mean(), 4)}')


# Results from upper tail
for a in alpha_lst:
    df = QIC_study.query(f"alpha=={a}")
    
    df_upper = df.query("tail == 'upper'")
    df_upper_VaR = df_upper.query("risk == 'VaR'")
    df_upper_ES = df_upper.query("risk == 'ES'")
    
    print(a)
    print('upper')
    print(f'{round(df_upper_VaR.qic.mean(),4)} & {round(df_upper_ES.qic.mean(),4)} & ~ & {round(df_upper_VaR.SE_theo.mean(), 4)} & {round(df_upper_ES.SE_theo.mean(),4)} & ~ & {round(df_upper_VaR.SE_garch.mean(), 4)} & {round(df_upper_ES.SE_garch.mean(), 4)}')
    
    
#%% Summary Statistics - Pearson

# INPUT
PEAR_results = pd.read_csv('results\studies\PEAR_GLOBAL.csv', index_col=0)

# ---------------------------------

freq_lst = list(set(PEAR_results.frequency))

for f in freq_lst:
    df = PEAR_results.query(f"frequency=='{f}'")
    
    print(f)
    print(df.cor.mean())
    print(df.SE_theo.mean())
    print(df.SE_garch.mean())
