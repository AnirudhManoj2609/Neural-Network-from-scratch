#!/usr/bin/env python
# coding: utf-8

# Import the libraries

# In[9]:


import numpy as np
import nnfs


# Initialize class

# In[10]:


class DenseLayer:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons)
        #each row correspond to the input and each column correspond to the neuron
        #this format prevents the need to take the transpose of the weights while taking dot with input vector
        self.biases = np.zeros((1,n_neurons))


# Training

# In[11]:


def forward(self,inputs):
    self.output = np.dot(inputs,self.weights) + self.biases
DenseLayer.forward = forward

