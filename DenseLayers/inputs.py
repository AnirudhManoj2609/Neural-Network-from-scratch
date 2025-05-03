#!/usr/bin/env python
# coding: utf-8

# Import libraries

# In[19]:


import sys
import importlib

# Add parent directory to sys.path so Python can see ReLu
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import and reload
import ReLu.relu
import SoftMax.softmax
import dense
importlib.reload(dense)
importlib.reload(ReLu.relu)
importlib.reload(SoftMax.softmax)
import nnfs
from nnfs.datasets import spiral_data
from dense import DenseLayer
from ReLu.relu import Activation_Relu
from SoftMax.softmax import Activation_Softmax
# In[20]:


X,y = spiral_data(samples=100,classes=3)

dense1 = DenseLayer(2,3)

activation1 = Activation_Relu()

dense2 = DenseLayer(3,3)
activation2 = Activation_Softmax()

dense1.forward(X)

activation1.forward(dense1.output)

dense2.forward(activation1.output)

activation2.forward(dense2.output)

print(activation2.output[:5])


