import numpy as np

class Activation_Relu:

    def forward(self,inputs):
        self.output = np.maximum(0,inputs)