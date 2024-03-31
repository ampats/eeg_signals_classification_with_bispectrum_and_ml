import numpy as np
import pandas as pd
from import_data3 import x as x
from import_data3 import y as y
from sklearn.feature_selection import SelectKBest, f_classif
from array import *

# choose k best features for all 3 classes
print("\nAll 3 classes")
xi = x
yi = y

# uncomment for each case

# # case 1: standing - slow walking
# print("\nstanding - slow walking")
# xi = x[0:30, :]

# # case 2: slow walking - running
# print("\nslow walking - running")
# xi = x[15:45, :]

# # case 3: standing - running
# print("\nstanding - running")
# xi = np.concatenate([x[0:15, :], x[30:45, :]], axis=0)

yi = y[0:30]

# select k-best features
xBest = SelectKBest(f_classif, k=10)
f_values = f_classif(xi, yi)
xBest.fit_transform(xi, yi)

# print f- and p- values for each feature
df = pd.DataFrame({'f-values':f_values[0],
                   'p-values':f_values[1],
                   })
#all 40 features
feature_names = np.array(["BispecPeak", "Freq of BispecPeak", "WCOB - f1m", "WCOB - f2m", "NBE", "NBSE", "J", "Skewness", "Kurtosis", "Variance", "MMOB", "Bispec-magnitude variability", "SOLA", "SOLADE_1", "SOLADE_2", "SOLAHE_3", "FOSM_1", "FOSM_2", "FOSM_3", "SOSM_1", "SOSM_2", "SOSM_3", "SumOfAmplitudesDE_1", "SumOfAmplitudesDE_2", "SumOfAmplitudesHE_3", "SSI_1", "SSI_2", "SSI_3", "RootMeanSquareDE_1", "RootMeanSquareDE_2", "RootMeanSquareHE_3", "VarDE_1", "VarDE_2", "VarHE_3", "V3orderDE_1", "V3orderDE_2", "V3orderHE_3", "LogDE_1", "LogDE_2", "LogHE_3"])
df.index = feature_names

print("\n")
print(df)
print("\n")
print("Selected features:\n")
print(df.index[xBest.get_support()])