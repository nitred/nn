'''
Author : Nitish Reddy Koripalli
Date : 14-11-2015
'''

import numpy as np
from ActivationFunction import ActivationFunction
from LayerStatistics import LayerStatistics

class MLP_BP:    
    DEBUG = False
    
    WEIGHTS_RANDOM = 1
    WEIGHTS_ZEROS = 2
    WEIGHTS_ONES = 3
    
    OUTPUT_CLASSIFICATION = 1
    OUTPUT_REGRESSION = 2
    
    TRAIN_CLASSIFICATION = 1
    TRAIN_REGRESSION = 2
    
    FORWARD_PASS_TRAIN = 1
    FORWARD_PASS_TEST = 2
    
    def __init__(self, network_design, initial_weights=WEIGHTS_RANDOM,
                 bias=False, learning_rate=1,
                 activation_alpha=0.1,
                 train_type=TRAIN_REGRESSION,
                 output_type=OUTPUT_REGRESSION):
        self.network_design = network_design
        self.initial_weights = initial_weights
        self.layer_stats_list = []
        self.bias = bias
        self.learning_rate = learning_rate
        self.activation_alpha = activation_alpha
        self.train_type = train_type
        self.output_type = output_type
        # initialize network
        self.__gen_network()
        
    def __gen_network(self):
        for i in range(len(self.network_design) - 1):
            layer_stats = LayerStatistics()
            self.layer_stats_list.append(layer_stats)
            # input layer flag
            if i == 0: layer_stats.is_input_layer = True
            else: layer_stats.is_input_layer = False
            # output layer flag
            if i == len(self.network_design) - 2: layer_stats.is_output_layer = True
            else: layer_stats.is_output_layer = False
                
            # learning rate
            layer_stats.LR = self.learning_rate
            
            # input - nothing to be initialized
            
            # weight matrix creation
            input_dim = self.network_design[i]
            layer_stats.input_dim = input_dim
            output_dim = self.network_design[i+1]
            layer_stats.output_dim = output_dim
            
            # initialize weight matrix, taking bias into consideration
            if self.initial_weights is self.WEIGHTS_RANDOM:
                if self.bias: weight_matrix = np.random.rand(input_dim + 1, output_dim)
                else: weight_matrix = np.random.rand(input_dim, output_dim)
            elif self.initial_weights is self.WEIGHTS_ONES:
                if self.bias: weight_matrix = np.ones((input_dim + 1, output_dim))
                else: weight_matrix = np.ones((input_dim, output_dim))
            else:
                if self.bias: weight_matrix = np.zeros((input_dim + 1, output_dim))
                else: weight_matrix = np.zeros((input_dim, output_dim))

            layer_stats.W = weight_matrix
            
            # local field - nothing to be initialized
            # activation function objects
            af_objs = np.empty((output_dim,1), dtype='object')
            for i in range(output_dim):
                af_obj = ActivationFunction(activation_alpha=self.activation_alpha)
                af_objs[i,0] = af_obj
            layer_stats.AF = af_objs
                
            # outputs/activations - nothing to be initialized
            # activation gradient - nothing to be initialized
            
    def print_weights(self):
        print "\n Network Weights"
        for i in range(len(self.layer_stats_list)):
            print "\nLayer :", i+1
            print self.layer_stats_list[i].W

    def __compute_activations(self, activation_functions, local_fields):
        activations = np.empty(local_fields.shape)
        for i in range(local_fields.shape[0]):
            activations[i,0] = activation_functions[i,0].get_activation(local_fields[i,0])
        return activations
    
    def __compute_activation_gradients(self, activation_gradients, local_fields):
        gradients = np.empty(local_fields.shape)
        for i in range(local_fields.shape[0]):
            gradients[i,0] = activation_gradients[i,0].get_gradient(local_fields[i,0])
        return gradients
    
    def __apply_output_type(self, layer_stats, fp_type):
        # if layer is output layer
        if layer_stats.is_output_layer:
            # if training and if training is supposed to use classification
            if fp_type is self.FORWARD_PASS_TRAIN and self.train_type is self.TRAIN_CLASSIFICATION:
                return np.array(layer_stats.A > 0.5, dtype='float')
            # if training and if training is supposed to use regression
            elif fp_type is self.FORWARD_PASS_TRAIN and self.train_type is self.TRAIN_REGRESSION:
                return layer_stats.A
            # if testing and output is supposed to use classification
            elif fp_type is self.FORWARD_PASS_TEST and self.output_type is self.OUTPUT_CLASSIFICATION:
                return np.array(layer_stats.A > 0.5, dtype='float')
            # if testing and output is suppsed to use regression
            else:
                return layer_stats.A
        # if layer is not output just use continous activation values
        else:
            return layer_stats.A
    
    def forward_pass(self, train_inputs, fp_type):
        if self.DEBUG:
            print "\nForward Pass:"
        for i in range(len(self.layer_stats_list)):
            if self.DEBUG:
                print "\nLayer :", i + 1
            layer_stats = self.layer_stats_list[i]
            # set train input only to input layer
            if layer_stats.is_input_layer:
                if self.bias: layer_stats.X = np.vstack((np.ones((1,1)), train_inputs))
                else: layer_stats.X = train_inputs
                    
                if self.DEBUG:
                    print "is input layer"
                    print "layer_stats.X.shape :", layer_stats.X.shape
            # calculate local fields
            layer_stats.V = np.dot(layer_stats.W.T, layer_stats.X)
            # calculate activations
            layer_stats.A = self.__compute_activations(layer_stats.AF, layer_stats.V)
            # calculate output
            layer_stats.Y = self.__apply_output_type(layer_stats, fp_type)
            
            if self.DEBUG:
                print "layer_stats.V.shape :", layer_stats.V.shape
                print "layer_stats.Y.shape :", layer_stats.Y.shape
                
            # set next layer inputs as current layer outputs
            if not layer_stats.is_output_layer:
                next_layer_stats = self.layer_stats_list[i+1]
                if self.bias: next_layer_stats.X = np.vstack((np.ones((1,1)),layer_stats.Y.copy()))
                else: next_layer_stats.X = layer_stats.Y.copy()
    
    def backward_pass(self, target):
        if self.DEBUG:
            print "\nBackward Pass"
        for i in reversed(range(len(self.layer_stats_list))):
            layer_stats = self.layer_stats_list[i]
            
            if self.DEBUG:
                print "\nLayer :", i
            
            if layer_stats.is_output_layer:
                if self.DEBUG:
                    print "Output Layer"
                # calculate the error
                layer_stats.E = target - layer_stats.Y
                # calculate activation error
                layer_stats.AE = target - layer_stats.A
                # calculate the activation gradient
                layer_stats.G = self.__compute_activation_gradients(layer_stats.AF, layer_stats.V)
                # calculate deltas
                layer_stats.D = layer_stats.E * layer_stats.G
                # update weights
                layer_stats.W += layer_stats.LR * np.dot(layer_stats.X, layer_stats.D.T)
                # compute Sigma DW (bias correction)
                if self.bias: layer_stats.DW = np.dot(layer_stats.W[1:,:], layer_stats.D)
                else: layer_stats.DW = np.dot(layer_stats.W, layer_stats.D)
                
                if self.DEBUG:
                    print "layer_stats.E.shape :", layer_stats.E.shape
                    print "layer_stats.G.shape :", layer_stats.G.shape
                    print "layer_stats.D.shape :", layer_stats.D.shape
                    print "layer_stats.W.shape :", layer_stats.W.shape
                    print "layer_stats.DW.shape :", layer_stats.DW.shape

            else:
                # calculate activation gradients
                layer_stats.G = self.__compute_activation_gradients(layer_stats.AF, layer_stats.V)
                # calculate deltas
                next_layer_stats = self.layer_stats_list[i+1]
                layer_stats.D = layer_stats.G * next_layer_stats.DW
                # update weights
                layer_stats.W += layer_stats.LR * np.dot(layer_stats.X, layer_stats.D.T) # outer product
                # calculate DW
                if self.bias: layer_stats.DW = np.dot(layer_stats.W[1:,:], layer_stats.D)
                else: layer_stats.DW = np.dot(layer_stats.W, layer_stats.D)
                    
                if self.DEBUG:
                    print "Non Output Layer"
                    print "layer_stats.G.shape :", layer_stats.G.shape
                    print "layer_stats.D.shape :", layer_stats.D.shape
                    print "layer_stats.W.shape :", layer_stats.W.shape
                    print "layer_stats.DW.shape :", layer_stats.DW.shape
    
    def train(self, inputs, targets, DEBUG=True):
        if DEBUG:
            print "Pre-Train Weights:\n", self.layer_stats_list[-1].W
        for i in range(inputs.shape[0]):
            input_ = inputs[i,:][:,np.newaxis]
            target_ = targets[i,:][:,np.newaxis]
            self.forward_pass(input_, self.FORWARD_PASS_TRAIN)
            self.backward_pass(target_)
            if DEBUG:
                print "\nTrain Iteration :", i+1
                #print "Output :", self.layer_stats_list[-1].Y
                #print "Weights:\n", self.layer_stats_list[-1].W
                print "Error:", self.layer_stats_list[-1].E
                #print "Activation Error:", self.layer_stats_list[-1].AE
            
    def test(self, inputs, DEBUG=True):
        if DEBUG:
            print "Testing"
        for i in range(inputs.shape[0]):
            input_ = inputs[i,:][:,np.newaxis]
            self.forward_pass(input_, self.FORWARD_PASS_TEST)
            
            if DEBUG:
                print "\nTest Iteration   :", i+1
                print "Test Input       :", input_.T
                print "Activation       :", self.layer_stats_list[-1].A
                print "Output           :", self.layer_stats_list[-1].Y