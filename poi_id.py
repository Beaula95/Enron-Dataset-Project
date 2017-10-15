#!/usr/bin/python

import sys
import matplotlib.pyplot
import pickle
import pandas as pd
import numpy as np
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit,train_test_split,cross_val_score

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
my_features_list=['poi','from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'shared_receipt_with_poi',
    'to_messages','bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'exercised_stock_options',
    'expenses',
    'loan_advances',
    'long_term_incentive',
    'other',
    'restricted_stock',
    'restricted_stock_deferred',
    'salary',
    'total_payments',
    'total_stock_value',
                 ]               

features_list = ["poi"]#features_list is a list where we add features by KBest Selection.
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
features = [ "salary","bonus"]
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus=point[1]
    matplotlib.pyplot.scatter( salary,bonus )
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
#Removing outliers from the data.
data_dict.pop('TOTAL',0)
data_dict.pop('The Travel Agency In the Park',0)

### Task 3: Create new feature(s)
def money_ratio_compute( expenses, salary ):
    result=0
    if expenses =='NaN': 
        return result
    if salary=='NaN': 
        return result
    result=float(expenses)/float(salary)
    return result
for n in data_dict:
    if data_dict[n]['poi']==1:
        poi_money_ratio=money_ratio_compute(data_dict[n]['expenses'],data_dict[n]['salary'])
        non_poi_money_ratio=0
    else:
        poi_money_ratio=0
        non_poi_money_ratio=money_ratio_compute(data_dict[n]['expenses'],data_dict[n]['salary'])
#Two features named poi_money_ratio for poi and non_poi_money_ratio are created.
    data_dict[n]['poi_money_ratio']=poi_money_ratio
    data_dict[n]['non_poi_money_ratio']=non_poi_money_ratio    
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest, f_classif
kbest = SelectKBest()
kbest = SelectKBest(k=10)
selected_features = kbest.fit_transform(features,labels)
feature_scores = kbest.scores_
features_selected=[my_features_list[i+1] for i in kbest.get_support(indices=True)]
features_scores_selected=[feature_scores[i] for i in kbest.get_support(indices=True)]
print 'Features selected by SelectKBest:'
print features_selected
print 'Feature score:'
print features_scores_selected
features_list.extend(features_selected)
print(features_list)
#Making new_list to add the new features to my_features_list to find feature score of new features.
new_list=my_features_list
new_list.append("poi_money_ratio")
new_list.append("non_poi_money_ratio")
data = featureFormat(my_dataset, new_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest, f_classif
kbest= SelectKBest()
kbest = SelectKBest(k=20)
selected_features = kbest.fit_transform(features,labels)
feature_scores = kbest.scores_
features_selected=[new_list[i+1] for i in kbest.get_support(indices=True)]
features_scores_selected=[feature_scores[i] for i in kbest.get_support(indices=True)]
print 'new Features selected by SelectKBest:'
print features_selected
print 'new Feature score:'
print features_scores_selected

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
features=scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
# Provided to give you a starting point. Try a variety of classifiers.
#Naive bayes
from sklearn.naive_bayes import GaussianNB
g_clf = GaussianNB()

#K-means clustering
from sklearn.cluster import KMeans
k_clf=KMeans(n_clusters=2,tol=0.001)
#Decision tree
from sklearn import tree
d_clf = tree.DecisionTreeClassifier()

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
estimators=[('reduce_dim',PCA()),('svm',SVC())]
pip_clf=Pipeline(estimators)
svc_clf=SVC(kernel='rbf',C=1000)
pca_clf=PCA(n_components=2)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
data = featureFormat(my_dataset,features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

skb = SelectKBest(k = 10)
g_clf =  Pipeline(steps=[('scaling',scaler),("SKB", skb), ("NaiveBayes", GaussianNB())])

pa = {'SKB__k': range(1,10)}
gs = GridSearchCV(g_clf, param_grid = pa, scoring = 'f1')
gs.fit(features, labels)

clf = gs.best_estimator_

print 'best algorithm'
print clf

g_clf.fit(features_train, labels_train)
pred = g_clf.predict(features_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred,labels_test)
accuracy = g_clf.score(features_test, labels_test)
print "accuracy GaussianNB",accuracy
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

pre = precision_score(labels_test, pred, labels=None, pos_label=1, average='binary', sample_weight=None)
print "pre",pre
rec = recall_score(labels_test, pred, labels=None, pos_label=1, average='binary', sample_weight=None)
print "rec",rec

from sklearn import tree
d_clf = tree.DecisionTreeClassifier()
parameters = {'criterion': ['gini', 'entropy'],
              'min_samples_split': [2, 10, 20],
               'max_depth': [None, 2, 5, 10],
               'min_samples_leaf': [1, 5, 10],
               'max_leaf_nodes': [None, 5, 10, 20]}
d_clf = GridSearchCV(d_clf, parameters)
d_clf.fit(features_train, labels_train)
pred_d= d_clf.predict(features_test)
from sklearn.metrics import accuracy_score
accuracy_d = accuracy_score(pred_d,labels_test)
accuracy_d = d_clf.score(features_test, labels_test)
print 'DecisionTree:'
print accuracy_d

pre_d = precision_score(labels_test, pred_d, labels=None, pos_label=1, average='binary', sample_weight=None)
print "pre",pre_d
rec_d = recall_score(labels_test, pred_d, labels=None, pos_label=1, average='binary', sample_weight=None)
print "rec",rec_d


sk_fold = StratifiedShuffleSplit(labels, 100, random_state = 42)
gs = GridSearchCV(g_clf, param_grid = pa, cv=sk_fold, scoring = 'f1')
gs.fit(features, labels)
clf = gs.best_estimator_

print 'best algorithm using strat_s_split'
print clf


test_classifier(clf, my_dataset, features_list)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)






