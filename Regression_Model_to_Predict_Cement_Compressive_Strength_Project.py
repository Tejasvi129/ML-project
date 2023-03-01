#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# import data
cement = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Concrete%20Compressive%20Strength.csv')
cement


# In[3]:


# view data
display(cement)


# In[4]:


# info of data
cement.info()


# In[5]:


# summary statistics
cement.describe()


# In[6]:


# check for missing value
cement.isna()


# In[17]:


# check for categories
cement.isna().sum()


# In[18]:


# visualize pairplot
sns.pairplot(cement)
plt.show()


# In[19]:


# columns name
corr = cement.corr()

sns.heatmap(corr, annot=True, cmap='Blues')
b, t = plt.ylim()
plt.ylim(b+0.5, t-0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


# In[20]:


# define y
y = cement.iloc[:,-1]    


# In[21]:


# define X
X = cement.iloc[:,:-1]  


# In[22]:


# split data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# In[23]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[24]:


# verify shape
from sklearn.linear_model import LinearRegression, Lasso, Ridge


# In[25]:


# select model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[26]:


# define X_new

X_new.shape


# In[ ]:


# predict for X_new
new_pred_class = logreg.predict(X_new)


# In[ ]:




