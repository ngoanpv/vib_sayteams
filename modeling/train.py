
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
# Import different models 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# Scoring function
from sklearn.metrics import roc_auc_score, roc_curve

training_data = pd.read_csv('vib_febatures.csv')

"""# Training"""



training_data.drop(columns = ['CLIENT_CREATE_DATE'], inplace=True)
X = training_data.drop(columns=['CUSTOMER_NUMBER', 'is_churned'], axis=1)
X = X.fillna(0)
y = training_data.is_churned
# y = y.fillna(False)
features_label = X.columns
forest = RandomForestClassifier (n_estimators = 5, random_state = 0, n_jobs = -1)
forest.fit(X, y)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
# for i in range(50):
#     print ("%2d) %-*s %f" % (i + 1, 30, features_label[i], importances[indices[i]]))

feat_importances = pd.Series(importances, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')

"""K-Nearest Neighbor (KNN)
Logistic Regression (LR)
AdaBoost
Gradient Boosting (GB)
RandomForest (RF)
"""

"""### Split dataset"""

# Splitting the dataset in training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Commented out IPython magic to ensure Python compatibility.
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import numpy as np
import pandas as pd
pd.set_option('precision', 3)
pd.options.mode.chained_assignment = None

import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'retina'

import seaborn as sns
sns.set_style('darkgrid')

from scipy.stats import chi2_contingency
from collections import Counter
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, recall_score, precision_score, auc, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# import scikitplot as skplt

label_size = 17

plt.rcParams['axes.labelsize'] = label_size
plt.rcParams['xtick.labelsize'] = label_size - 2
plt.rcParams['ytick.labelsize'] = label_size - 2
plt.rcParams['axes.titlesize'] = label_size
plt.rcParams['legend.fontsize'] = label_size - 2

random_state = 42
scoring_metric = 'recall'
comparison_dict = {}
comparison_test_dict = {}

print ('Libraries Loaded!')

# !pip install scikitplot

before_sm = Counter(y_train)
print(before_sm)

over = SMOTE(sampling_strategy = 'auto', random_state = 42)
X_train, y_train = over.fit_resample(X_train, y_train)

after_sm = Counter(y_train)
print(after_sm)

random_state = 42

def plot_continuous(feature):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 4))
    
    sns.distplot(df_remained[feature], bins = 15, color = colors[0], label = 'Remained', hist_kws = dict(edgecolor = 'firebrick', linewidth = 1), ax = ax1, kde = False)
    sns.distplot(df_churned[feature], bins = 15, color = colors[1], label = 'Churned', hist_kws = dict(edgecolor = 'firebrick', linewidth = 1), ax = ax1, kde = False)
    ax1.set_title('{} distribution - Histogram'.format(feature))
    ax1.set_ylabel('Counts')
    ax1.legend()

    sns.boxplot(x = 'Exited', y = feature, data = train_df, palette = colors, ax = ax2)
    ax2.set_title('{} distribution - Box plot'.format(feature))
    ax2.set_xlabel('Status')
    ax2.set_xticklabels(['Remained', 'Churned'])

    plt.tight_layout();
    
    
def plot_categorical(feature):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 4))
    sns.countplot(x = feature, hue = 'Exited', data = train_df, palette = colors, ax = ax1)
    ax1.set_ylabel('Counts')
    ax1.legend(labels = ['Retained', 'Churned'])
    
    sns.barplot(x = feature, y = 'Exited', data = train_df, palette = colors2 , ci = None, ax = ax2)
    ax2.set_ylabel('Churn rate')
    
    if (feature == 'HasCrCard' or feature == 'IsActiveMember'):
        ax1.set_xticklabels(['No', 'Yes'])
        ax2.set_xticklabels(['No', 'Yes'])
    
    plt.tight_layout();
    
def plot_learning_curve(estimator, estimator_name, X, y, cv = None, train_sizes = np.linspace(0.1, 1.0, 5)):
                 
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv = cv, n_jobs = -1, 
                                                            train_sizes = train_sizes, scoring = 'accuracy')
    
    train_scores_mean, train_scores_std = np.mean(train_scores, axis = 1), np.std(train_scores, axis = 1)
    test_scores_mean, test_scores_std = np.mean(test_scores, axis = 1), np.std(test_scores, axis = 1)
            
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha = 0.1, color = 'dodgerblue')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha = 0.1, color = 'darkorange')
    
    plt.plot(train_sizes, train_scores_mean, color = 'dodgerblue', marker = 'o', linestyle = '-', label = 'Training Score')
    plt.plot(train_sizes, test_scores_mean, color = 'darkorange', marker = 'o', linestyle = '-', label = 'Cross-validation Score')
    plt.title(estimator_name)
    plt.xlabel('Training Examples')
    plt.ylabel('Accuracy Score')
    plt.legend(loc = 'best')
            
    plt.tight_layout();
    
def plot_conf_mx(cm, classifier_name, ax):
    sns.heatmap(cm, annot = True, cmap = 'Blues', annot_kws = {'fontsize': 24}, ax = ax)
    ax.set_title('{}'.format(classifier_name))
    ax.set_xlabel('Predicted Label')
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(['Remained', 'Churned'])
    ax.set_ylabel('True Label') 
    ax.set_yticks([0.25, 1.25])
    ax.set_yticklabels(['Remained', 'Churned']);
    
def plot_feature_imp(classifier, classifier_name, color, ax):

    importances = pd.DataFrame({'Feature': X_train.columns,
                                'Importance': np.round(classifier.best_estimator_.feature_importances_, 3)})
    importances = importances.sort_values('Importance', ascending = True).set_index('Feature')
    importances.plot.barh(color = color, edgecolor = 'firebrick', legend = False, ax = ax)
    ax.set_title(classifier_name)
    ax.set_xlabel('Importance');
    
def clf_performance(classifier, classifier_name, classifier_name_abv):
    print('\n', classifier_name)
    print('-------------------------------')
    print('   Best Score ({}): '.format(scoring_metric) + str(np.round(classifier.best_score_, 3)))
    print('   Best Parameters: ')
    for key, value in classifier.best_params_.items() :
        print ('      {}: {}'.format(key, value))
    
    y_pred_pp = cross_val_predict(classifier.best_estimator_, X_train, y_train, cv = 5, method = 'predict_proba')[:, 1]
    y_pred = y_pred_pp.round()
    
    cm = confusion_matrix(y_train, y_pred, normalize = 'true')
    
    fpr, tpr, _ = roc_curve(y_train, y_pred_pp)
    
    comparison_dict[classifier_name_abv] = [accuracy_score(y_train, y_pred), 
                                            precision_score(y_train, y_pred),
                                            recall_score(y_train, y_pred),
                                            roc_auc_score(y_train, y_pred_pp),
                                            fpr, tpr]    

    fig, ax = plt.subplots(figsize = (5, 4))
    
    plot_conf_mx(cm, '', ax)    
    plt.tight_layout();
    
def test_func(classifier, classifier_name):
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, normalize = 'true')
    
    comparison_test_dict[classifier_name] = [accuracy_score(y_test, y_pred), 
                                             precision_score(y_test, y_pred),
                                             recall_score(y_test, y_pred)]
    
    plt.title(classifier_name)
    sns.heatmap(cm, annot = True, annot_kws = {'fontsize': 18}, cmap = 'Blues')
    plt.xlabel('Predicted Label')
    plt.xticks([0.5, 1.5], ['Remained', 'Churned'])
    plt.ylabel('True Label') 
    plt.yticks([0.2, 1.4], ['Remained', 'Churned']);
    
print ('Functions defined!')

lr = LogisticRegression(random_state = random_state)

param_grid = {'max_iter' : [100],
              'penalty' : ['l1', 'l2'],
              'C' : [0.001, 0.01, 0.1, 1, 10],
              'solver' : ['lbfgs', 'liblinear']}

clf_lr = GridSearchCV(lr, param_grid = param_grid, scoring = scoring_metric, 
                      cv = 5, verbose = False, n_jobs = -1)

best_clf_lr = clf_lr.fit(X_train, y_train)

clf_performance(best_clf_lr, 'Logistic Regression', 'LR')

svc = SVC(probability = True, random_state = random_state)
param_grid = tuned_parameters = [{'kernel': ['rbf'],
                                  'gamma': ['scale', 'auto'],
                                  'C': [.1, 1, 2]},
                                 {'kernel': ['linear'], 
                                  'C': [.1, 1, 10]}]
clf_svc = GridSearchCV(svc, param_grid = param_grid, scoring = scoring_metric, cv = 5, verbose = False, n_jobs = -1)
best_clf_svc = clf_svc.fit(X_train, y_train)
clf_performance(best_clf_svc, 'Support Vector Classifier', 'SVC')

rf = RandomForestClassifier(random_state = random_state)
param_grid = {'n_estimators': [50],
              'criterion': ['entropy', 'gini'],
              'bootstrap': [True],
              'max_depth': [6],
              'max_features': ['auto','sqrt'],
              'min_samples_leaf': [2, 3, 5],
              'min_samples_split': [2, 3, 5]}
                                  
clf_rf = GridSearchCV(rf, param_grid = param_grid, scoring = scoring_metric, cv = 5, verbose = False, n_jobs = -1)
best_clf_rf = clf_rf.fit(X_train, y_train)
clf_performance(best_clf_rf, 'Random Forest Classifier', 'RF')

gbc = GradientBoostingClassifier(random_state = random_state)
param_grid = {'n_estimators': [600],
              'subsample': [0.66, 0.75],
              'learning_rate': [0.001, 0.01],
              'max_depth': [3],                       # default=3
              'min_samples_split': [5, 7],
              'min_samples_leaf': [3, 5],
              'max_features': ['auto', 'log2', None],
              'n_iter_no_change': [20],
              'validation_fraction': [0.2],
              'tol': [0.01]}
                                  
clf_gbc = GridSearchCV(gbc, param_grid = param_grid, scoring = scoring_metric, cv = 5, verbose = False, n_jobs = -1)
best_clf_gbc = clf_gbc.fit(X_train, y_train)
clf_performance(best_clf_gbc, 'Gradient Boosting Classifier', 'GBC')

best_clf_gbc.best_estimator_.n_estimators_

xgb = XGBClassifier(random_state = random_state)

param_grid = {'n_estimators': [50],
              'learning_rate': [0.001, 0.01],
              'max_depth': [3, 4],                # default=6
              'reg_alpha': [1, 2],
              'reg_lambda': [1, 2],
              'subsample': [0.5, 0.75],
              'colsample_bytree': [0.50, 0.75],
              'gamma': [0.1, 0.5, 1],
              'min_child_weight': [1]}

clf_xgb = GridSearchCV(xgb, param_grid = param_grid, scoring = scoring_metric, cv = 5, verbose = False, n_jobs = -1)
best_clf_xgb = clf_xgb.fit(X_train, y_train)
clf_performance(best_clf_xgb, 'XGBoost Classifier', 'XGBC')

lgbmc = LGBMClassifier(random_state = random_state)

param_grid = {'max_depth': [5],
              'num_leaves': [5, 10],
              'learning_rate': [0.001, 0.01],
              'n_estimators': [200],
              'feature_fraction': [0.5],
              'min_child_samples': [5, 10],
              'reg_alpha': [0.1, 0.5],
              'reg_lambda': [0.1, 0.5]} 

clf_lgbmc = GridSearchCV(lgbmc, param_grid = param_grid, verbose = False,
                         scoring = scoring_metric, cv = 5, n_jobs = -1)

best_clf_lgbmc = clf_lgbmc.fit(X_train, y_train)
clf_performance(best_clf_lgbmc, 'LGBMClassifier', 'LGBMC')

fig, ax = plt.subplots(3, 3, figsize = (15, 15))

for i in range(len(estimators)):
    plt.subplot(3, 3, i + 1)
    plot_learning_curve(estimators[i][1], estimators[i][0], X_train, y_train)
    
plt.tight_layout()
ax[2,1].set_axis_off()
ax[2,2].set_axis_off();

comparison_df.plot(kind = 'bar', figsize = (10, 5), fontsize = 12, color = ['#5081DE', '#A7AABD', '#D85870', '#424656'])

plt.legend(loc = 'upper center', ncol = len(comparison_df.columns), bbox_to_anchor = (0.5, 1.11))
plt.xticks(rotation = 0)
plt.yticks([0, 0.4, 0.8])

plt.axhline(y = 0.75, color = 'red', linestyle = '--')
plt.text(x = -0.45, y = 0.77, s = '0.75', size = label_size + 2, color = 'red');

comparison_matrix = {}
for key, value in comparison_dict.items():
    comparison_matrix[str(key)] = value[0:4]

comparison_df = pd.DataFrame(comparison_matrix, index = ['Accuracy', 'Precision', 'Recall', 'AUC']).T
comparison_df.style.highlight_max(color = 'indianred', axis = 0)

color_ = ['steelblue', 'darkgray', 'cadetblue', 'bisque']

fig = plt.subplots(2, 2, figsize = (10, 8))

for i, (name, clf) in enumerate(zip(['RF', 'GB', 'XGB', 'LGBM'], 
                                    [best_clf_rf, best_clf_gbc, best_clf_xgb, best_clf_lgbmc])):
    
    ax = plt.subplot(2, 2, i + 1)
    plot_feature_imp(clf, name, color_[i], ax)
    plt.ylabel('')
    
plt.tight_layout();

colors = ['steelblue', 'seagreen', 'black', 'darkorange', 'purple', 'firebrick', 'slategrey']

fig = plt.figure(figsize = (8, 5))

for index, key in enumerate(comparison_dict.keys()):
    auc, fpr, tpr = comparison_dict[key][3], comparison_dict[key][4], comparison_dict[key][5]
    plt.plot(fpr, tpr, color = colors[index], label = '{}: {}'.format(key, np.round(auc, 3)))

plt.plot([0, 1], [0, 1], 'k--', label = 'Baseline')

plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.xticks([0, 0.25, 0.5, 0.75, 1])
plt.ylabel('False Positive Rate')
plt.yticks([0, 0.25, 0.5, 0.75, 1])
plt.legend(fontsize = 14);

print ('Soft Voting:')

y_pred = cross_val_predict(tuned_voting_soft, X_train, y_train, cv = 5, method = 'predict_proba')
skplt.metrics.plot_cumulative_gain(y_train, y_pred)

plt.plot([0.5, 0.5], [0, 0.8], color = 'firebrick')
plt.plot([0.0, 0.5], [0.8, 0.8], color = 'firebrick')
plt.text(0.15, 0.81, '80%', size = label_size, color = 'firebrick');

"""### Test set"""

tuned_voting_soft.fit(X_train, y_train)
fig = plt.subplots(7, 1, figsize = (5, 25))

for i, (name, clf) in enumerate(zip(['LR', 'SVC', 'RF', 'GB', 'XGB', 'LGBM', 'SVot'],
                                    [best_clf_lr.best_estimator_, best_clf_svc.best_estimator_, best_clf_rf.best_estimator_, best_clf_gbc.best_estimator_, best_clf_xgb.best_estimator_, best_clf_lgbmc.best_estimator_, tuned_voting_soft])):
    
    plt.subplot(7, 1, i + 1)
    test_func(clf, name)
    
plt.tight_layout();

"""# Models Set 2

KNN
"""

# Initialization of the KNN
knMod = KNeighborsClassifier(n_neighbors = 5, weights = 'uniform', algorithm = 'auto', leaf_size = 30, p = 2,
                             metric = 'minkowski', metric_params = None)
# Fitting the model with training data 
knMod.fit(X_train, y_train)

"""Logistic Regression"""

# Initialization of the Logistic Regression
lrMod = LogisticRegression(penalty = 'l2', dual = False, tol = 0.0001, C = 1.0, fit_intercept = True,
                            intercept_scaling = 1, class_weight = None, 
                            random_state = None, solver = 'liblinear', max_iter = 100,
                            multi_class = 'ovr', verbose = 2)
# Fitting the model with training data 
lrMod.fit(X_train, y_train)

"""AdaBoost model"""

# Initialization of the AdaBoost model
adaMod = AdaBoostClassifier(base_estimator = None, n_estimators = 200, learning_rate = 1.0)
# Fitting the model with training data 
adaMod.fit(X_train, y_train)

"""GradientBoosting model"""

# Initialization of the GradientBoosting model
gbMod = GradientBoostingClassifier(loss = 'deviance', n_estimators = 200)
# Fitting the model with training data 
gbMod.fit(X_train, y_train)

from xgboost import XGBClassifier, plot_importance

clf  = XGBClassifier(max_depth = 10,random_state = 10, n_estimators=220, eval_metric = 'auc', min_child_weight = 3,
                    colsample_bytree = 0.75, subsample= 0.9)

clf.fit(X_train, y_train)

# Compute the model accuracy on the given test data and labels
xgb_acc = knMod.score(X_test, y_test)
# Return probability estimates for the test data
test_labels = knMod.predict_proba(np.array(X_test.values))[:,1]
# Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
xgb_roc_auc = roc_auc_score(y_test, test_labels , average = 'macro', sample_weight = None)

xgb_roc_auc, xgb_acc

"""Random Forest model"""

# Initialization of the Random Forest model
rfMod = RandomForestClassifier(n_estimators=100, criterion='gini')
# Fitting the model with training data 
rfMod.fit(X_train, y_train)

"""## Testing the baseline model"""

# Compute the model accuracy on the given test data and labels
knn_acc = knMod.score(X_test, y_test)
# Return probability estimates for the test data
test_labels = knMod.predict_proba(np.array(X_test.values))[:,1]
# Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
knn_roc_auc = roc_auc_score(y_test, test_labels , average = 'macro', sample_weight = None)

# Compute the model accuracy on the given test data and labels
lr_acc = lrMod.score(X_test, y_test)
# Return probability estimates for the test data
test_labels = lrMod.predict_proba(np.array(X_test.values))[:,1]
# Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
lr_roc_auc = roc_auc_score(y_test, test_labels , average = 'macro', sample_weight = None)

# Compute the model accuracy on the given test data and labels
ada_acc = adaMod.score(X_test, y_test)
# Return probability estimates for the test data
test_labels = adaMod.predict_proba(np.array(X_test.values))[:,1]
# Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
ada_roc_auc = roc_auc_score(y_test, test_labels , average = 'macro')

# Compute the model accuracy on the given test data and labels
gb_acc = gbMod.score(X_test, y_test)
# Return probability estimates for the test data
test_labels = gbMod.predict_proba(np.array(X_test.values))[:,1]
# Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
gb_roc_auc = roc_auc_score(y_test, test_labels , average = 'macro')

# Compute the model accuracy on the given test data and labels
rf_acc = rfMod.score(X_test, y_test)
# Return probability estimates for the test data
test_labels = rfMod.predict_proba(np.array(X_test.values))[:,1]
# Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
rf_roc_auc = roc_auc_score(y_test, test_labels , average = 'macro')

models = ['KNN', 'Logistic Regression', 'AdaBoost', 'GradientBoosting', 'Random Forest']
accuracy = [knn_acc, lr_acc, ada_acc, gb_acc, rf_acc]
roc_auc = [knn_roc_auc, lr_roc_auc, ada_roc_auc, gb_roc_auc, rf_roc_auc]
d = {'accuracy': accuracy, 'roc_auc': roc_auc}
df_metrics = pd.DataFrame(d, index = models)
df_metrics

xgb_roc_auc

fpr_knn, tpr_knn, _ = roc_curve(y_test, knMod.predict_proba(np.array(X_test.values))[:,1])
fpr_lr, tpr_lr, _ = roc_curve(y_test, lrMod.predict_proba(np.array(X_test.values))[:,1])
fpr_ada, tpr_ada, _ = roc_curve(y_test, adaMod.predict_proba(np.array(X_test.values))[:,1])
fpr_gb, tpr_gb, _ = roc_curve(y_test, gbMod.predict_proba(np.array(X_test.values))[:,1])
fpr_rf, tpr_rf, _ = roc_curve(y_test, rfMod.predict_proba(np.array(X_test.values))[:,1])

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, test_labels)

# Plot the roc curve
plt.figure(figsize = (12,6), linewidth= 1)
plt.plot(fpr_knn, tpr_knn, label = 'KNN Score: ' + str(round(knn_roc_auc, 5)))
plt.plot(fpr_lr, tpr_lr, label = 'LR score: ' + str(round(lr_roc_auc, 5)))
plt.plot(fpr_ada, tpr_ada, label = 'AdaBoost Score: ' + str(round(ada_roc_auc, 5)))
plt.plot(fpr_gb, tpr_gb, label = 'GB Score: ' + str(round(gb_roc_auc, 5)))
plt.plot(fpr_rf, tpr_rf, label = 'RF score: ' + str(round(rf_roc_auc, 5)))
# plt.plot(fpr_xgb, tpr_xgb, label = 'XGB score: ' + str(round(xgb_roc_auc, 5)))
plt.plot([0,1], [0,1], 'k--', label = 'Random guessing: 0.5')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve ')
plt.legend(loc='best')
plt.show()

# Import the cross-validation module
from sklearn.model_selection import cross_val_score

# Function that will track the mean value and the standard deviation of the accuracy
def cvDictGen(functions, scr, X_train = X, y_train = y, cv = 5):
    cvDict = {}
    for func in functions:
        cvScore = cross_val_score(func, X_train, y_train, cv = cv, scoring = scr)
        cvDict[str(func).split('(')[0]] = [cvScore.mean(), cvScore.std()]
    
    return cvDict

mod = [knMod, lrMod, adaMod, gbMod, rfMod]
cvD = cvDictGen(mod, scr = 'roc_auc')
cvD

'GradientBoostingClassifier': [0.9122906201028478, 0.0019474190475162537],
  'RandomForestClassifier': [0.9122891811217654, 0.0017918665655620569]}

# Import methods
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

adaHyperParams = {'n_estimators': [10,50,100,200,420]}

gridSearchAda = RandomizedSearchCV(estimator = adaMod, param_distributions = adaHyperParams, n_iter = 5,
                                   scoring = 'roc_auc')
gridSearchAda.fit(X_train, y_train)

# Display the best parameters and the score
gridSearchAda.best_params_, gridSearchAda.best_score_

rfMod.predict_proba(np.array(X_test.values))[:,1]

from sklearn.metrics import classification_report
print(classification_report(y_test,  rfMod.predict_proba(np.array(X_test.values))[:,1] > 0.5))

from sklearn.metrics import classification_report
print(classification_report(y_test,  gbMod.predict_proba(np.array(X_test.values))[:,1] > 0.5))

# Import the log transformation method
from sklearn.preprocessing import FunctionTransformer, StandardScaler
transformer = FunctionTransformer(np.log1p)
scaler = StandardScaler()
X_train_1 = np.array(X_train)
#X_train_transform = transformer.transform(X_train_1)
X_train_transform = scaler.fit_transform(X_train_1)

bestGbModFitted_transformed = gbMod.fit(X_train_transform, y_train)
bestEFModFitted_transformed = rfMod.fit(X_train_transform, y_train)

cvDictbestpara_transform = cvDictGen(functions = [bestGbModFitted_transformed, bestEFModFitted_transformed],
                                     scr='roc_auc')
cvDictbestpara_transform

# For the test set
X_test_1 = np.array(X_test)
#X_test_transform = transformer.transform(X_test_1)
X_test_transform = scaler.fit_transform(X_test_1)

test_labels=bestGbModFitted_transformed.predict_proba(np.array(X_test_transform))[:,1]
roc_auc_score(y_test,test_labels , average = 'macro', sample_weight = None)

# Import the voting-based ensemble model
from sklearn.ensemble import VotingClassifier

# Initialization of the model
votingMod = VotingClassifier(estimators=[('gb', bestGbModFitted_transformed), 
                                         ('rf', bestEFModFitted_transformed)],
                                         voting = 'soft', weights = [2,1])
# Fitting the model
votingMod = votingMod.fit(X_train_transform, y_train)

test_labels=votingMod.predict_proba(np.array(X_test_transform))[:,1]
votingMod.score(X_test_transform, y_test)

# The roc_auc score
roc_auc_score(y_test, test_labels , average = 'macro', sample_weight = None)

print(classification_report(y_test,  test_labels > 0.5))

test_labels

from sklearn.metrics import matthews_corrcoef
matthews_corrcoef(y_test,  test_labels > 0.5)

bestGbModFitted_transformed.feature_importances_

feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')

feat_importances = pd.Series(bestGbModFitted_transformed.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')

feat_importances = pd.Series(gbMod.feature_importances_, index=X_train.columns)
feat_importances.nlargest(20).plot(kind='barh')

feat_importances = pd.Series(rfMod.feature_importances_, index=X_train.columns)
feat_importances.nlargest(20).plot(kind='barh')

