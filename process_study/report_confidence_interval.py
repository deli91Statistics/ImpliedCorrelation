# -----------------------------------------------------
# SUMMARY STATISTICS
# -----------------------------------------------------
#
# This module loads a QIC/PEARSON study and creates a 
# latex table that contains all confidence intervals.
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



# Was soll ich tun? Alles zusammen in eine Tabelle
# Mit Boxplot Visualisieren?