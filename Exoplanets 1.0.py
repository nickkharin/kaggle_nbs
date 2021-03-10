#!/usr/bin/env python
# coding: utf-8

# In[116]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import  metrics
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, plot_confusion_matrix


# In[117]:


X_train = pd.read_csv('exoTrain.csv')
y_train = X_train['LABEL']
X_train = X_train.drop(columns=['LABEL'], axis=1)
X_test =pd.read_csv('exoTest.csv')
y_test = X_test['LABEL']
X_test = X_test.drop(columns=['LABEL'], axis=1)


# In[118]:


X_train.head()


# I chose a robust scaler for standardization

# In[129]:


scl = RobustScaler()
scl.fit(X_train)
X_train_scl = scl.transform(X_train)
scl.fit(X_test)
X_test_scl = scl.transform(X_test)


# ## Model: Gradient Boosting

# In[135]:


GB = GradientBoostingClassifier()
GB.fit(X_train_scl, y_train)
prediction_GB=GB.predict(X_test_scl)
train_score_GB = GB.score(X_train_scl, y_train)
test_score_GB = GB.score(X_test_scl, y_test)
print(f"Gradient Boosting train score: {train_score_GB}")
print(f"Gradient Boosting test score: {test_score_GB}")
print(classification_report(y_test, prediction_GB))


# ## Model: Decision Tree

# In[131]:


DT = DecisionTreeClassifier()
DT.fit(X_train_scl, y_train)
prediction_DT=DT.predict(X_test_scl)
train_score_DT = DT.score(X_train_scl, y_train)
test_score_DT = DT.score(X_test_scl, y_test)
print(f"Decision Tree train score: {train_score_DT}")
print(f"Decision Tree test score: {test_score_DT}")
print('Decision Tree Classifier')
print(classification_report(y_test, prediction_DT))


# ## Model: AdaBoost

# In[133]:


AB = AdaBoostClassifier()
AB.fit(X_train_scl, y_train)
prediction_AB=AB.predict(X_test_scl)
train_score_AB = AB.score(X_train_scl, y_train)
test_score_AB = AB.score(X_test_scl, y_test)
print(f"AdaBoost train score: {train_score_AB}")
print(f"Adaboost test score: {test_score_AB}")
print(classification_report(y_test, prediction_AB))


# ## Model: Random Forest

# In[134]:


RF = RandomForestClassifier()
RF.fit(X_train_scl, y_train)
prediction_RF=RF.predict(X_test_scl)
train_score_RF = RF.score(X_train_scl, y_train)
test_score_RF = RF.score(X_test_scl, y_test)
print(f"RF train score: {train_score_RF}")
print(f"RF test score: {test_score_RF}")
print(classification_report(y_test, prediction_RF))


# ## Model: Bagging

# In[128]:


BG = BaggingClassifier()
BG.fit(X_train, y_train)
prediction_BG=BG.predict(X_test)
train_score_BG = BG.score(X_train, y_train)
test_score_BG = BG.score(X_test, y_test)
print(f"BG train score: {train_score_BG}")
print(f"BG test score: {test_score_BG}")
print(classification_report(y_test, prediction_BG))


# ## Model compairing

# I used some kinds of models: Gradient Boosting, Decision Tree, AdaBoost, Random Forest and Bagging. I chose them for better understanding of how they work. As we can see on the diagram, they perform good results. 
# I will try some experiments with another models soon.

# In[139]:


data = {'Gradient Boosting': {'Train': train_score_GB, 'Test': test_score_GB},
        'Decision Tree': {'Train': train_score_DT, 'Test': test_score_DT},
        'AdaBoost': {'Train': train_score_AB, 'Test': test_score_AB},
        'Random Forest': {'Train': train_score_RF, 'Test': test_score_RF},
        'Bagging': {'Train': train_score_BG, 'Test': test_score_BG}}
df = pd.DataFrame(data)
df = df.T
df ['sum'] = df.sum(axis=1)
df.sort_values('sum', ascending=False)[['Test','Train']].plot.bar() 
plt.ylabel('Score')


# In[ ]:




