
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("ggplot")


from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import scale

from scipy.stats import randint
from scipy.stats import skew


from keras.layers import Dense
from keras.models import Sequential

import xgboost as xgb

import math as math

import seaborn as sns

import h2o
from h2o.automl import H2OAutoML


# In[24]:


cancer = pd.read_csv("data.csv")
cancer.head()


# In[4]:


cancer.diagnosis.value_counts()

# We should make the diagnosis value into a classification problem, that means replacing both values with 0 & 1
# In[25]:


cancer = cancer.replace(("B" , "M") , (0,1))


# In[26]:


cancer = cancer.drop(["id" , "Unnamed: 32"] , axis = 1)


# In[27]:


cancer.diagnosis = cancer.diagnosis.astype("category")


# In[42]:


cancer.corr().style.background_gradient().set_precision(2)


# In[60]:


def plot_scatter(x):
    
    plt.scatter(cancer["diagnosis"],cancer[x],alpha=0.5)
    plt.title(cancer[x].astype(str).name)
    plt.xlabel("Diagnosis")
    plt.ylabel(cancer[x].astype(str).name)
    plt.show()
    
    return " "


# In[61]:


for n in cancer.columns:
    print(plot_scatter(n))

# After turning the diagnosis column into a category and visualizing the relationships of each variable with the diagnosis we can now perform our prediction
# In[28]:


X_cancer = cancer.drop(["diagnosis"] , axis = 1)
y_cancer = cancer.diagnosis

X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, test_size = 0.3, random_state=42)


# In[41]:


def plot_num(x):
    
    X_cancer[x].plot.kde()
    plt.title(X_cancer[x].astype(str).name)
    plt.show()
    return ""

for n in X_cancer:
    print(plot_num(n))

# We can create a line plot to show the most common frequencies of each variable
# In[ ]:


# Logistic Regression


# In[39]:


logreg = LogisticRegression()
logreg.fit(X_train , y_train)
y_pred = logreg.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test , y_pred))
print(accuracy_score(y_test , y_pred))

# Decision Tree
# In[38]:


tree = DecisionTreeClassifier()

tree.fit(X_train , y_train)
y_pred = tree.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test , y_pred))
print(accuracy_score(y_test , y_pred))

# Decision Tree + Random Search CV
# In[36]:


param_dist = {'max_depth' : [3,None],
             'max_features' : np.arange(1,9),
             'min_samples_leaf':np.arange(1,9),
             "criterion" :["gini" , "entropy"]}

tree = DecisionTreeClassifier()
tree_cv = RandomizedSearchCV(tree , param_dist , cv = 5 , verbose = 1)

tree_cv.fit(X_train , y_train)
y_pred_2 = pd.DataFrame(tree_cv.predict(X_test))
print(tree_cv.best_params_)
print(tree_cv.best_score_)
print(classification_report(y_test , y_pred_2))

# Our best accuracy is the logistic regression with 96% accuracy.