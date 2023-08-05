#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import pandas as pd
import joblib


# In[2]:


model = joblib.load(r'\Model\WBY_Prediction.joblib')


# In[10]:


"""
# Wild Blueberry Yield Prediction Application
Input values for the following **features** below:
"""


# In[4]:


def user_input_features():
    clonesize = st.slider('Clonesize',10.0,40.0,0.0)
    bumbles = st.slider('Bumbles',0.0,0.585,0.0)
    andrena = st.slider('Andrena',0.0,0.75,0.0)
    osmia = st.slider('Osmia',0.0,0.75,0.0)
    AverageOfUpperTRange = st.slider('AverageOfUpperTRange',58.2,79.0,0.0)
    AverageOfLowerTRange = st.slider('AverageOfLowerTRange',41.2,55.9,0.0)
    AverageRainingDays = st.slider('AverageRainingDays',0.06,0.56,0.0)
    data = {'clonesize':clonesize,
           'bumbles':bumbles,
           'andrena':andrena,
           'osmia':osmia,
           'AverageOfUpperTRange':AverageOfUpperTRange,
           'AverageOfLowerTRange':AverageOfLowerTRange,
           'AverageRainingDays':AverageRainingDays}
    features = pd.DataFrame(data,index=[0])
    return features


# In[5]:


df = user_input_features()


# In[6]:


st.subheader('User Input Parameters')
st.write(df)


# In[7]:


st.subheader('Prediction')
st.write(np.power(10, model.predict(df)))


# In[ ]:





# In[ ]:




