#!/usr/bin/env python
# coding: utf-8

# Import libraries

# In[ ]:


import numpy as np


# initialize input batch for training

# In[ ]:


inputs = [[1.5,6.7,2.2,1.3],
          [5.6,3.4,6.2,10.1],
          [2.3,4.5,1.6,23.1]] #3*4 matrix


# Initialize the weights and bias

# In[ ]:


weights = [[0.2,-0.4,0.8,0.7],
           [-0.24,-0.9,0.21,0.76],
           [0.25,-0.56,-0.12,0.43]]
bias = [2.0,3.0,0.5]


# Perform the dot operation

# In[ ]:


output = np.dot(inputs,np.array(weights).T) + bias


# Display output

# In[ ]:


print(output)

