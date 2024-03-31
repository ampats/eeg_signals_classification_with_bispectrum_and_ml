#leave-one-subject-out with AUC and ROC curve - for data from all channels

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from import_data3 import x as x
from import_data3 import y as y
from import_data3 import nFeat as nFeat
from import_data3 import nSampl as nSampl
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import svm
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

# choose model 

# SVM Classifier
# model = svm.SVC(kernel='linear', class_weight='balanced', probability=True, random_state=0)
# model.fit(x, y)
# model: Random Forest
model = RandomForestClassifier(n_estimators=500, random_state=1)
# model: DTree
# model = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=1)
# SoftMax
# model = LogisticRegression(multi_class='multinomial',
#                           solver='newton-cg',
#                           C = 100.0,
#                           penalty='l2',
#                           random_state=1)

# # Building the pipeline (for Softmax, KNN or SVM)
# pipe = Pipeline(steps = [('std', StandardScaler()), ('model', model)])

# ROC - AUC & LOSO CV

# data from all electrodes and all subs except for 1, for 2 classes each time
xSample = np.zeros((252,nFeat))
ySample = []
nBoot = 100 # number of bootstrap samples

# case 1: standing - slow walking
x1 = x[0:270, :]

# case 2: slow walking - running
x2 = x[135:405, :]

# case 3: standing - running
x3 = np.concatenate([x[0:135, :], x[270:405, :]], axis=0)


xAll = [x1, x2, x3]
yAll = y[0:270]

# calculating AUC for each pair of classes

# change xii depending on the pair of classes you want to check for (x1, x2 or x3)
xii = x1

yAll = y[0:270]

xTrList = []
yTrList = []
xTeList = []
yTeList = []

tprs = []
base_fpr = np.linspace(0, 1, 101)
plt.figure(figsize=(15, 15))
plt.axes().set_aspect('equal', 'datalim')
aucs = []
all_y = []
all_probs=[]
sampleBoot = np.zeros((nSampl*2*9,2))

for l in range(0,127,9):  #nSampl: nChan*(nSub-1)

    deleted_rows = [l,l+1,l+2,l+3,l+4,l+5,l+6,l+7,l+8,l+nSampl*9,l+nSampl*9+1,l+nSampl*9+2,l+nSampl*9+3,l+nSampl*9+4,l+nSampl*9+5,l+nSampl*9+6,l+nSampl*9+7,l+nSampl*9+8]
    x_train = np.delete(xii,deleted_rows,axis=0)
    y_train = np.delete(yAll,deleted_rows,axis=0)

    x_test = np.array([xii[l:l+9], xii[l+nSampl*9:l+nSampl*9+9]])
    x_test = np.reshape(x_test,(18,40))
    y_test = np.array([yAll[l:l+9], yAll[l+nSampl*9:l+nSampl*9+9]])
    y_test = y_test.astype(np.float32)
    y_test = np.reshape(y_test,(18,1))
    
    all_y.append(y_test)
    all_probs.append(model.fit(x_train, y_train).predict_proba(x_test)[:,1])

all_y = np.array(all_y)
all_y = np.reshape(all_y,(270,1))
#all_y = np.reshape(all_y,(30,1))

all_probs = np.array(all_probs)

all_probs = np.reshape(all_probs,(270,1))

predictions = np.concatenate([all_y, all_probs], axis=1)
predictions = predictions[predictions[:,0].argsort()]


for i in range(nBoot):

    sampleBoot[0:135, 0] = predictions[0:135,0]
    sampleBoot[0:135, 1] = random.choices(predictions[0:135,1], k=135)
    sampleBoot[135:270, 0] = predictions[135:270,0]
    sampleBoot[135:270, 1] = random.choices(predictions[135:270,1], k=135)

    all_y = sampleBoot[:,0]
    all_probs = sampleBoot[:,1]
    fpr, tpr, thresholds = roc_curve(all_y,all_probs)
    roc_auc = auc(fpr,tpr)
    aucs.append(roc_auc)
    plt.title("ROC curve: Standing & Slow walking")
    tpr = np.interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)
tprs = np.array(tprs)
mean_tprs = tprs.mean(axis=0)
std = tprs.std(axis=0)
tprs_upper = np.minimum(mean_tprs + std, 1)
tprs_lower = mean_tprs - std
mean_auc = auc(base_fpr, mean_tprs)
plt.plot(base_fpr, mean_tprs, 'b', label= 'Mean AUC = %0.2f' % mean_auc)

#ci
alphaVal = 0.95
p = ((1.0-alphaVal)/2.0) * 100
lower = max(0.0, np.percentile(aucs, p))
p = (alphaVal+((1.0-alphaVal)/2.0)) * 100
upper = min(1.0, np.percentile(aucs, p))

plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3, label='CI: (%0.2f, %0.2f)' % (lower, upper))
plt.legend(loc='best')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
