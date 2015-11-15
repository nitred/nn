'''
Author : Nitish Reddy Koripalli
Date : 14-11-2015
'''

from sympy import Symbol, diff, exp

class ActivationFunction:
    def __init__(self, activation_alpha=1):
        self.v = Symbol('v')
        self.alpha = activation_alpha
        self.activation_function = 1 / (1 + exp(-self.alpha*self.v))
        self.activation_function_diff = diff(self.activation_function)
    
    def get_activation(self, local_field):
        return self.activation_function.evalf(subs={self.v:local_field})
        
    def get_gradient(self, local_field):
        return self.activation_function_diff.evalf(subs={self.v:local_field})