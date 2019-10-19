# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from patsy import dmatrices


# In[2]:

data = pd.read_csv("../data/gradient_descent_data.csv")


# In[3]:

data.head()


# In[9]:

data.dtypes


# #standard operation to make one hot encoding for non-digit features

# In[11]:

y, X = dmatrices("left~satisfaction_level+last_evaluation+number_project+average_montly_hours+time_spend_company+Work_accident+promotion_last_5years+C(sales)+C(salary)"
    , data
    , return_type = "dataframe")


# In[12]:

X.head()


# In[13]:

y.head()


# #make y, X numpy matrix and one d array for matrix multiplication

# In[18]:

X = np.asmatrix(X)
y = np.ravel(y)
type(X), type(y)


# #make all the features in [0, 1] range

# In[22]:

for i in range(1, X.shape[1]):
    colMin = X[:, i].min()
    colMax = X[:, i].max()
    X[:, i] = (X[:, i]  - colMin)/(colMax - colMin)


# In[24]:

X


# #start doing gradient decent

# In[25]:

np.random.seed(1)


# In[26]:
# learning rate
alpha = 1 


# In[27]:

# randomly create a beta as lr assignment
beta = np.random.randn(X.shape[1])   


# In[44]:

losses = []
error_rates = []
derives = []
for T in range(1000):
    
    # define loss function
    prob = 1.0 / (1  + np.exp(-np.matmul(X, beta)))      # p = 1/1 + e ^ -bx
    prob = np.array(prob).ravel()                        # turn p into one d array to zip with y
    pyzip = list(zip(prob, y))
    loss = -sum(np.log(p) if yi == 1 else np.log(1-p) for p, yi in pyzip )/len(y)
    
    # caluclate error rate, which is the REAL loss
    error_rate = 0.0
    for i in range(len(y)):
        if ((prob[i] > 0.5 and y[i] == 0) or (prob[i] <= 0  and y[i] == 1)) :
            error_rate += 1
    error_rate /= len(y)
    
    # derivation ( gradient decent)
    deriv = np.zeros(X.shape[1])
    for i in range(len(y)):
        #deriv += (prob[i] - y[i]) * X[i]
        deriv += np.asarray(X[i,:]).ravel() * (prob[i] - y[i])   
        
    deriv /= len(y)
    
    
    # debug
    losses.append(loss)
    error_rates.append(error_rate)
    derives.append(deriv[0])
    
    if T % 50 == 0 :
        print("T = " + str(T) + ", loss = " + str(loss) + ", error rate = " + str(error_rate))
        print("deriv = " + str(deriv))
    # follow the oppsite direction of derivation to change the beta
    beta -= alpha * deriv 
    


# In[50]:

type(losses)


# In[53]:

x = range(1000)
plt.plot(x, losses)


# In[54]:

plt.plot(x, error_rates)


# In[52]:

plt.plot(x, derives)


# In[ ]:

# the best way to get the best arguments is to find the best accuracy on validation set

