#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot
import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv("Social_Network_Ads.csv")
data.head()


# In[3]:


data.isna().sum()


# In[4]:


X=data.iloc[ : , :-1].values
y=data.iloc[ : , -1].values


# In[5]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[6]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[7]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier




# In[8]:


# def logistic_regression_model(x_train,y_train):
#     log_reg = LogisticRegression()
#     log_reg.fit(x_train, y_train)
#     return log_reg


# In[9]:


# def Xgboost_model(x_train,y_train):
#     n_estimators = [50, 100, 150, 200]
#     max_depth = [2, 4, 6, 8]
#     learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
#     subsample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
#     colsample_bytree = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
#     min_child_weight = [1, 2, 3, 4, 5]
#     gamma = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#     params = {
#         "n_estimators": n_estimators,
#         "max_depth": max_depth,
#         "learning_rate": learning_rate,
#         "subsample": subsample,
#         "colsample_bytree": colsample_bytree,
#         "min_child_weight": min_child_weight,
#         "gamma": gamma,
#     }
#     xgb = XGBClassifier()
#     clf = RandomizedSearchCV(xgb, params, cv=5, n_jobs=-1)
#     clf.fit(x_train,y_train)
#     return clf


# In[10]:


# def svm_model(x_train,y_train):  
#     C = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#     gamma = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#     params = {
#         "C": C,
#         "gamma": gamma
#     }
#     svm = SVC(probability=True)
#     clf = RandomizedSearchCV(svm, params, cv=5, n_jobs=-1)
#     clf.fit(x_train,y_train)
#     return clf


# In[11]:


def random_forest_model(x_train,y_train):
    n_estimators = [50, 100, 150, 200]
    max_depth = [2, 4, 6, 8]
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
    }
    rf = RandomForestClassifier()
    clf = RandomizedSearchCV(rf, params, cv=5, n_jobs=-1)
    clf.fit(x_train, y_train)
    return clf



# In[12]:


# def KNN_model(x_train,y_train): 
#     n_neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#     weights = ['uniform', 'distance']
#     algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
#     leaf_size = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#     p = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#     params = {
#         "n_neighbors": n_neighbors,
#         "weights": weights,
#         "algorithm": algorithm,
#         "leaf_size": leaf_size,
#         "p": p,
#     }
#     knn = KNeighborsClassifier()
#     clf = RandomizedSearchCV(knn, params, cv=5, n_jobs=-1)
#     clf.fit(x_train,y_train)
#     return clf


# In[13]:


# lr=logistic_regression_model(X_train,y_train)


# In[14]:


# xgb=Xgboost_model(X_train,y_train)


# In[15]:


# svm=svm_model(X_train,y_train)


# In[16]:


rf=random_forest_model(X_train,y_train)


# In[17]:


# knn=KNN_model(X_train,y_train)


# In[18]:


from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.metrics import roc_curve, roc_auc_score




# In[19]:


# y_pred_lr = lr.predict(X_test)


# In[20]:


# y_pred_xgb = xgb.predict(X_test)


# In[21]:


# y_pred_svm = svm.predict(X_test)


# In[22]:


y_pred_rf = rf.predict(X_test)


# In[23]:


# y_pred_knn = knn.predict(X_test)


# In[24]:


# y_pred_proba_lr = lr.predict_proba(X_test)


# In[25]:


# y_pred_proba_xgb = xgb.predict_proba(X_test)


# In[26]:


# y_pred_proba_svm = svm.predict_proba(X_test)


# In[27]:


y_pred_proba_rf = rf.predict_proba(X_test)


# In[28]:


# y_pred_proba_knn = knn.predict_proba(X_test)


# In[29]:


def evaluate_model(y_test, y_pred, y_pred_proba):
    metrics = {
        "Accuracy Score": accuracy_score(y_test, y_pred),
        "Precision Score (micro)": precision_score(y_test, y_pred, average='micro'),
        "Precision Score (macro)": precision_score(y_test, y_pred, average='macro'),
        "Recall Score (micro)": recall_score(y_test, y_pred, average='micro'),
        "Recall Score (macro)": recall_score(y_test, y_pred, average='macro'),
        "F1 Score (micro)": f1_score(y_test, y_pred, average='micro'),
        "F1 Score (macro)": f1_score(y_test, y_pred, average='macro'),
        "Log Loss": log_loss(y_test, y_pred_proba)
    }
    return metrics




# In[30]:


# evaluate_model(y_test,y_pred_lr,y_pred_proba_lr)


# In[31]:


# evaluate_model(y_test,y_pred_xgb,y_pred_proba_xgb)


# In[32]:


# evaluate_model(y_test,y_pred_svm,y_pred_proba_svm)


# In[33]:


evaluate_model(y_test,y_pred_rf,y_pred_proba_rf)


# In[34]:


# evaluate_model(y_test,y_pred_knn,y_pred_proba_knn)


# In[ ]:





# In[ ]:




