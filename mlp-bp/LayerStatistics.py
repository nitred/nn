'''
Author : Nitish Reddy Koripalli
Date : 14-11-2015
'''

class LayerStatistics:
    def __init__(self):
        # general
        self.input_dim = None
        self.output_dim = None
        self.is_output_layer = False
        self.is_input_layer = False
        # inputs
        self.X = None # inputs
        self.W = None # weight matrix
        # outputs
        self.V = None # local fields
        self.AF = None # activation function object
        self.A = None # activation values (continuous values)
        self.Y = None # outputs
        self.G = None # gradients
        # updation
        self.AE = None # activation errors
        self.E = None # errors
        self.D = None # deltas
        self.DW = None # sigma of product of deltas and input weights
        self.LR = None # learning rate