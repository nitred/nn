'''
Author : Nitish Reddy Koripalli
Date : 14-11-2015
'''

from MLP_BP import MLP_BP
import numpy as np

if __name__ == '__main__':
    train_data = np.genfromtxt("./train/XOR.txt", delimiter=",", skip_header=1)
    test_data = np.genfromtxt("./test/XOR.txt", delimiter=",", skip_header=1)
    
    mlp_bp = MLP_BP(network_design=[2,2,1],
              initial_weights=MLP_BP.WEIGHTS_RANDOM,
              bias=True,
              learning_rate=1,
              train_type=MLP_BP.TRAIN_REGRESSION,
              output_type=MLP_BP.OUTPUT_CLASSIFICATION,
              activation_alpha=1)
    
    # Training
    TRAIN_ITERATIONS = 1000 # (iterations of the same file)
    
    for i in range(TRAIN_ITERATIONS):
        train_data_ = train_data.copy()
        np.random.shuffle(train_data_)
        train_inputs_ = train_data_[:,:2]
        train_targets_ = train_data_[:,2][:,np.newaxis]
        
        mlp_bp.train(train_inputs_, train_targets_, DEBUG=False)
    
    # Testing
    test_input_ = test_data[:,:2]
    mlp_bp.test(test_input_, DEBUG=True)