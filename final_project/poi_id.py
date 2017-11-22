#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt
import numpy as np

# Select desired features

features_list = ['poi', 
    'shared_receipt_with_poi','salary', 'total_payments', 
   'loan_advances', 'bonus', 'deferred_income', 
   'total_stock_value', 'exercised_stock_options', 
    'salary_and_bonus'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)
    
###  Remove outliers
###  Create new feature(s)
### Store to my_dataset for easy export below.

features = ["salary", "bonus"]
data_dict.pop('TOTAL', 0)
data_dict.pop('SKILLING JEFFREY K', 0)
for key in data_dict:
	if key == 'SKILLING JEFFREY K':
		print(data_dict[key])

data = featureFormat(data_dict, features)

def print_k_values(num):
    
    """Use to determine best k value"""

    for k in range(num):
        selector = SelectKBest(f_classif, k=k)
        selector.fit(predictors, labels_train)
        scores = -np.log10(selector.pvalues_)
        print(scores)

for key, value in data_dict.items():
	
	# create new feature

	if value['bonus'] == 'NaN':
		value['bonus'] = 0.
	if value['salary'] == 'NaN':
		value['salary'] = 0.
			
	value['salary_and_bonus'] = value['salary'] + value['bonus']
	#print(value['salary_and_bonus'])

my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Look for best features

from sklearn.feature_selection import SelectKBest, f_classif

predictors = features_train

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(predictors, labels_train)
#print_k_values(11)

# Get the raw p-values for each feature, and transform from p-values into scores

scores = -np.log10(selector.pvalues_)
print(scores)

from sklearn.ensemble import RandomForestClassifier

# create a pipeline to see which model performs best

"""
# this pipeline was used to determine the best model
# I left this code in for reference
# to test this code, replace RandomForestClassifier() with
# the variable pipe in the grid search and comment out the other param_grid

pipe = Pipeline([("preprocessing", StandardScaler()), ('classifier', SVC())])

param_grid = [
    {'classifier': [SVC()], 'preprocessing': [StandardScaler(), None, MinMaxScaler()],
    'classifier__gamma':[0.001,0.01,0.1,1,10,100],
    'classifier__C':[0.001,0.01,0.1,1,10,100]},
    
    {'classifier': [RandomForestClassifier(n_estimators=5)],
    'preprocessing':[None], 'classifier__max_features': [1,2,3],
    'classifier__criterion': ('gini', 'entropy'),
    'classifier__min_samples_split': [2,4,6],
    'classifier__min_samples_leaf': [1,2,3],
    'classifier__max_depth': [None,1, 2,4]},

    {'classifier': [DecisionTreeClassifier()], 'preprocessing':[None],
    'classifier__criterion': ('gini', 'entropy'),
    'classifier__splitter': ('best', 'random'),
    'classifier__max_depth': [None,1,2,4],
    'classifier__min_samples_split': [2,4,6],
    'classifier__min_samples_leaf': [1,2,3],
    'classifier__max_features': [None, 1,2,3] }

]
"""

# After all the work done above, the best model i've tried so far for tester.py was a
# regular decision tree with the default parameters. Unbelievable.

#clf = GridSearchCV(pipe, param_grid, cv=5)
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred_grid = clf.predict(features_test)
#confusion = confusion_matrix(labels_test, pred_grid)

#print("Best params:\n{}\n".format(clf.best_params_))
#print("Best cross-validation score:{:.2f}".format(clf.best_score_))
print("Test-set score: {:.2f}".format(clf.score(features_test,labels_test)))
print(classification_report(labels_test, pred_grid, target_names = ["non poi", "poi"]))
#print("Confusion matrix:\n{}".format(confusion))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
