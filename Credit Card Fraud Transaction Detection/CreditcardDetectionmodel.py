#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


#loading dataset into pandas data frame


# In[3]:


dt=pd.read_csv('creditcard.csv')


# In[4]:


dt


# In[5]:


dt.describe() #provide statistical analysis of dataset


# In[6]:


dt[dt['Class']==1]


# In[7]:


dt.isna().sum()


# # There are no null values in the dataset

# In[8]:


dt.head(10)


# In[9]:


dt.tail()


# # description of dataset
# #Time- Represents the time in seconds after the first transaction has occured
# 
# #v1,,,,...v28  represents features of each transaction. these details are sensitive so they didn't give more info about these features
# 
# #Amount- represents transcaction amount is us dollars
# 
# #Class- 0-Legit or norml transaction ,1- fraud transaction

# In[10]:


dt.info()


# In[11]:


dt['Class'].value_counts() #Highly unbalanced dataset  more than 99 percent are normal transactions


# In[12]:


normal_tr=dt[dt['Class']==0] #seperating normal and fraud transactions
fraud_tr=dt[dt['Class']==1]
normal_tr.count()


# In[13]:


fraud_tr.count()


# In[14]:


print(normal_tr.shape)
print(fraud_tr.shape)


# In[15]:


normal_tr['Amount'].describe()


# In[16]:


fraud_tr['Amount'].describe()


# In[17]:


dt.groupby('Class').mean()


# #Under Sampling- Build a sample datasset from original dataset which contain similar distribution of fraudelnt and legit transations

# In[18]:


normal_sample=normal_tr.sample(492) #randomly take 492 samples from 284000 samples


# In[19]:


normal_sample


# In[20]:


newdt=pd.concat([normal_sample,fraud_tr],axis=0)


# In[21]:


newdt.shape


# In[22]:


newdt.head()


# In[23]:


newdt.tail()


# In[24]:


newdt.groupby('Class').mean()


# In[25]:


newdt['Class'].value_counts()


# In[26]:


x=newdt.drop('Class',axis=1)
y=newdt['Class']


# In[27]:


scores=[]
for i in range(100):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=i)
    lr=LogisticRegression()
    lr.fit(x_train,y_train)
    y_pred=lr.predict(x_test)
    scores.append(accuracy_score(y_test,y_pred))


# In[28]:


np.argmax(scores)


# In[29]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)


# In[30]:


x


# In[31]:


y


# In[32]:


x_train


# In[33]:


x_test


# In[56]:


logr=LogisticRegression()
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()


# In[57]:


logr.fit(x_train,y_train)
rf.fit(x_train,y_train)
dt.fit(x_train,y_train)


# In[58]:


y_pred=logr.predict(x_test)
y_pred_rf=rf.predict(x_test)
y_pred_dt=dt.predict(x_test)


# In[59]:


score=accuracy_score(y_test,y_pred)
score_rf=accuracy_score(y_test,y_pred_rf)
score_dt=accuracy_score(y_test,y_pred_dt)


# In[60]:


print("Testing Accuracy is :",score)
print("Testing Accuracy is :",score_rf) #Best performing model on given dataset both and training ans testing
print("Testing Accuracy is :",score_dt)


# In[61]:


train_score=logr.predict(x_train)
train_score_rf=rf.predict(x_train)
train_score_dt=dt.predict(x_train)


# In[62]:


acc2_score=accuracy_score(y_train,train_score)
acc2_score_rf=accuracy_score(y_train,train_score_rf)
acc2_score_dt=accuracy_score(y_train,train_score_dt)


# In[63]:


print("Training Accuracy is:",acc2_score)
print("Training Accuracy is:",acc2_score_rf)
print("Training Accuracy is:",acc2_score_dt)


# In[64]:


newdt.to_csv('newdata.csv')


# In[65]:


k=newdt.to_csv('dt2.csv')


# In[66]:


k


# In[ ]:




