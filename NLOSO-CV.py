#nested leave-one-subject-out CV for data from all channels

import numpy as np
from numpy import mean
from numpy import absolute
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from import_data3 import x as x
from import_data3 import y as y
from import_data3 import nSampl as nSampl
from sklearn import svm
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# x (feature matrix) and y (target) data
# subjects is an array that associates each row of x with a subject

subjects = np.array([1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,11,12,12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,13,14,14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15,15,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,11,12,12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,13,14,14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15,15,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8,8,9,9,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,10,11,11,11,11,11,11,11,11,11,12,12,12,12,12,12,12,12,12,13,13,13,13,13,13,13,13,13,14,14,14,14,14,14,14,14,14,15,15,15,15,15,15,15,15,15])
# if I have only data from 1 electrode/channel for each subject:
# subjects = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) 
n_subjects = len(np.unique(subjects))
n_leave_out = 27  # Number of rows to leave out for each subject in each fold
# n_leave_out = 3 # if I have only data from 1 electrode for each subject
k_best_features = 20  # Number of top features to select with SelectKBest

# Initialize a list to store results (e.g., evaluation metrics)
results = []

# Outer loop for Leave-One-Subject-Out Cross-Validation
# Excluding data from subject 15 since we will use it for model evaluation
for i in range(n_subjects-1): 
    train_indices = np.where((subjects != i+1) & (subjects != 15))[0]  # Indices of rows for subjects not equal to i 
    test_indices = np.where((subjects == i+1) & (subjects != 15))[0]  # Indices of rows for subject i
        
    # defining the model (choose the one you want and comment the rest)

    # model = RandomForestClassifier(random_state=1) # Random Forest
    model = LogisticRegression(multi_class='multinomial', 
                         solver='newton-cg',
                        random_state=1) # Softmax
    # model = svm.SVC(random_state=1) #SVM
    # model = DecisionTreeClassifier(random_state=1) #DecisionTree


    # Building the pipeline (for Softmax, KNN or SVM)
    pipe = Pipeline(steps = [('std', StandardScaler()), ('model', model)])

    # Setting up the parameter grids 

    # param_grid = [{'n_estimators': [10, 100, 500, 1000, 10000]}] # Random Forest
    param_grid = [{'model__penalty': ['l2'],
                'model__C': np.power(10., np.arange(-4, 4))}] # Softmax
    # param_grid = [{'model__kernel': ['rbf'],
    #             'model__C': np.power(10., np.arange(-4, 4)),
    #             'model__gamma': np.power(10., np.arange(-5, 0))},
    #            {'model__kernel': ['linear'],
    #             'model__C': np.power(10., np.arange(-4, 4))}] # SVM
    # param_grid = [{'max_depth': list(range(1, 10)) + [None],
    #             'criterion': ['gini', 'entropy']}] #Decision Tree

    # Inner loop for K-Fold Cross-Validation on the training data
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    gcv = GridSearchCV(estimator=pipe,
                       param_grid=param_grid,
                       scoring='accuracy',
                       n_jobs=-1,
                       cv=cv_inner,
                       verbose=0,
                       refit=True)

    for train, val in cv_inner.split(train_indices, y[train_indices]):
        train_subset = train_indices[train]
        val_subset = train_indices[val]

        # Get the corresponding rows from x and y
        X_train = x[train_subset]
        y_train = y[train_subset]
        X_val = x[val_subset]
        y_val = y[val_subset]

        # # Uncomment to apply SelectKBest feature selection
        # selector = SelectKBest(score_func=f_classif, k=k_best_features)
        # X_train = selector.fit_transform(X_train, y_train)
        # X_val = selector.transform(X_val)

        # Train and evaluate model on the current fold
        gcv.fit(X_train, y_train) # run inner loop hyperparam tuning
        print('\n        Best ACC (avg. of inner test folds) %.2f%%' % (gcv.best_score_ * 100))
        print('        Best parameters:', gcv.best_params_)        

        # Append the evaluation metric to the results list
        results.append(gcv.best_estimator_.score(X_val, y_val))
        print('        ACC (on outer test fold) %.2f%%' % (results[-1]*100))

# Calculate the final performance estimate, e.g., mean score from the results list
print('\n    Outer Loop:');'[ ]'
print('        ACC %.2f%% +/- %.2f' % 
            (np.mean(results) * 100, np.std(results) * 100))

# uncomment after running the above and determine best model based on the Nested CV accuracy scores
# select a hyperparameters for the model based on regular k-fold on the whole training set

#using data from sub 15 to evaluate model
train_indices = np.where(subjects != 15)[0]  # Indices of rows for subjects not equal to i
print(train_indices)
test_indices = np.where(subjects == 15)[0]  # Indices of rows for subject i
print(test_indices)

x_train = x[train_indices]
y_train = y[train_indices]
x_test = x[test_indices]
y_test = y[test_indices] 

# # Uncomment to apply SelectKBest feature selection
# selector = SelectKBest(score_func=f_classif, k=k_best_features)
# x_train = selector.fit_transform(x_train, y_train)
# x_test = selector.transform(x_test)

gcv_model_select = GridSearchCV(estimator=pipe,
                                param_grid=param_grid,
                                scoring='accuracy',
                                n_jobs=-1,
                                cv=cv_inner,
                                verbose=1,
                                refit=True)

gcv_model_select.fit(x_train, y_train)
print('Best CV accuracy: %.2f%%' % (gcv_model_select.best_score_*100))
print('Best parameters:', gcv_model_select.best_params_)
    
# training the best model to the whole training set using the settings above
train_acc = accuracy_score(y_true=y_train, y_pred=gcv_model_select.predict(x_train))
test_acc = accuracy_score(y_true=y_test, y_pred=gcv_model_select.predict(x_test))

print('Training Accuracy: %.2f%%' % (100 * train_acc))
print('Test Accuracy: %.2f%%' % (100 * test_acc))