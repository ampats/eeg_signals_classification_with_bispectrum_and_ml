import csv
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

#load feature matrix
nFeat = 40
reader = csv.reader(open("data_noBad_chan1.csv", "r"), delimiter=",") # replace 1 with 2-9 to import data for each from the 9 channels
data = list(reader)
data = np.array(data).astype("float")
x = data[:,0:nFeat]
x = x.astype(np.float32)

#labels matrix (0=standing, 1=slow walking, 2=running)
nSampl = 15 
y = data[:,nFeat]
y = y.transpose()
labels = ["standing","slow walking", "running"]

# uncomment to select k best features instead of all of them
# x = SelectKBest(f_classif, k=31).fit_transform(x, y)
# nFeat = x.shape[1]