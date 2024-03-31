from matplotlib import pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from import_data3 import x as X
from import_data3 import y as y
from import_data3 import nFeat as nFeat
#from import_data3 import nSampl as nSampl
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler

# data loaded
random_state = np.random.RandomState(0)
n_classes = len(np.unique(y))
n_sampl = X.shape[0]
(
    X_train,
    X_test,
    y_train,
    y_test,
) = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

# choose model

# model = svm.SVC(kernel='linear', class_weight='balanced', probability=True, random_state=0) # model: SVM Classifier
# model = RandomForestClassifier(n_estimators=500, random_state=1)
# y_score = model.fit(X_train, y_train).predict_proba(X_test) # model: Random Forest
model = DecisionTreeClassifier(criterion='gini', max_depth=2, random_state=1)
y_score = model.fit(X_train, y_train).predict_proba(X_test) # model: DTree
# model = LogisticRegression(multi_class='multinomial',
#                           solver='newton-cg',
#                           C = 100.0,
#                           penalty='l2',
#                           random_state=1) # SoftMax

# # Building the pipeline (uncomment for Softmax, KNN or SVM)
# pipe = Pipeline(steps = [('std', StandardScaler()), ('model', model)])
# y_score = pipe.fit(X_train, y_train).predict_proba(X_test)

#  binarize the target by one-hot-encoding in a OvR fashion
label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_test)

# ROC curve using micro-averaged OvR
RocCurveDisplay.from_predictions(
    y_onehot_test.ravel(),
    y_score.ravel(),
    name="micro-average OvR",
    color="darkorange",
    plot_chance_level=True,
)
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Micro-averaged One-vs-Rest\nReceiver Operating Characteristic")
plt.legend()
plt.show()

# ROC curve showing a specific class: uncomment only the class you want to choose
# comment lines 50-62

# case 1: standing
class_of_interest = "standing"
class_id = 0

# # case 2: slow walking
# class_of_interest = "slow walking"
# class_id = 1

# # case 3: running
# class_of_interest = "running"
# class_id = 2

# RocCurveDisplay.from_predictions(
#     y_onehot_test[:, class_id],
#     y_score[:, class_id],
#     name=f"{class_of_interest} vs the rest",
#     color="darkorange",
#     plot_chance_level=True,
# ) 
# plt.axis("square")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.legend()
# plt.show()