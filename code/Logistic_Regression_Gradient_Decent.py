
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from patsy import dmatrices


# In[2]:

data = pd.read_csv("../data/gradient_descent_data.csv") 


# In[4]:

data.head()


# In[9]:

type(data.left[0])


# In[10]:

data.rename(columns={"average_montly_hours":"average_monthly_hours"}, inplace = True)


# In[11]:

data.head()


# # Convert data to two Dataframe's with left as y, other columns as X, also encode the categorial features

# In[16]:

y, X = dmatrices("left~satisfaction_level+last_evaluation+number_project+average_monthly_hours+time_spend_company+Work_accident+promotion_last_5years+C(sales)+C(salary)", data, return_type="dataframe")


# In[17]:

type(X), type(y)


# Convert X, y as numpy matrix and ndArray

# In[18]:

X = np.asmatrix(X)
y = np.ravel(y)


# In[19]:

type(X), type(y)


# # Data nomalization of X to ([0,1))

# In[22]:

for i in range(1, X.shape[1]):
    xmin = X[:, i].min()
    xmax = X[:, i].max()
    X[:, i] = (X[:,i] - xmin)/(xmax-xmin)


# In[24]:

X


# Create an array of randome number, the same size of the number of features

# In[26]:

np.random.seed(1)
alpha = 1
# create a initial beta, which is the arguments of the Logistic Regression model
beta = np.random.randn(X.shape[1])
beta                        


# # Definition of gradient decent

# In[34]:

# do 500 steps of gradient decent
for T in range(500):
    # use the initial model(aguments) creatd above to get the first y
    prob = np.array(1./(1+np.exp(-np.matmul(X, beta)))).ravel()
    # zip the calculated y with real y
    prob_y = list(zip(prob, y))
    # calculate loss with Logistic Regression cross entropy loss function
    loss = -sum([np.log(p) if y == 1 else np.log(1-p) for p, y in prob_y])/len(y)
    
    # calculate error rate
    error_rate = 0
    for i in range(len(y)):
        if ( prob[i] > 0.5 and y[i] == 0 ) or ( prob[i] <= 0.5 and y[i] == 1 ) :
            error_rate += 1
    error_rate /= len(y)
     
    # print progress once 5 steps
    if T % 5 == 0:
        print( "T = " + str(T) + " Loss = " + str(loss) + " error_rate = " + str(error_rate))
    
    # calculate Derivative
    # df/db = sum[Xi * (yi' - yi)]
    deriv = np.zeros(X.shape[1])
    for i in range(len(y)):
        deriv += np.asarray(X[i,:]).ravel() * (prob[i] - y[i])
    deriv /= len(y)
    
    # update the beta on the oppsite direction of derivative
    beta = beta - deriv


# ending Loss = 0.43154384232365245 error_rate = 0.2058803920261351

# In[ ]:



