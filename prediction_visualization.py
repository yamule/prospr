#!/usr/bin/env python
# coding: utf-8

# ## Visualization tools for ProSPr Distance Predictions
# Dr. Dennis Della Corte       
# Brigham Young University, Utah   
# 2019-10-30

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl  
import pandas as pd

def load(fileName): 
    fileObject2 = open(fileName, 'rb') 
    modelInput = pkl.load(fileObject2) 
    fileObject2.close() 
    return modelInput


# #### Modify these identifiers to match the run with which you obtained predictions:

# In[2]:


#example
domain_id = 'T1016-D1'
network = '1'
stride = '25'


# In[3]:


#load and display prediction data
prediction = load('./'+domain_id+'_'+str(network)+'_'+str(stride)+'_predictions.pkl')

plt.imshow(-np.argmax(prediction[1:,:,:],axis=0))
plt.suptitle('ProSPr '+domain_id+' Prediction', fontsize=14)
plt.title('Network '+network+', Stride '+stride, fontsize=10)
plt.xlabel('Residue i')
plt.ylabel('Residue j')

