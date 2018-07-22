import build_gan as GANs
import time 
import os 
from code import utilities
import tensorflow as tf 

from collections import defaultdict
from collections import Counter
import random 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
############ DATA PREPROCESSING ###################

# num_examples = 400 
inc = 10 

def standardizing(bucket): 
    
    for key in bucket:
        
        temp_train = np.concatenate(bucket[key], axis=0)
        scaler = preprocessing.StandardScaler().fit(temp_train)
        
        temp_list = [] 
        
        for value in bucket[key]:
            
            temp_list.append(scaler.transform(value))
            
        bucket[key] = temp_list    
        
#         print(bucket[key])

    return bucket 

def train_test_scenario_one(bucket):
    
    test_data_x = [] 
    test_data_y = [] 
    train_data_x = [] 
    train_data_y = [] 
    test_list_y = [] 
        
    for key in bucket:
        
        random.shuffle(bucket[key])
        if key[3] == 'H' or key[3] == 'n' :
            
            value = bucket[key].pop(0)
            test_data_x.append(value)
            test_data_y.append(key[3])

        for element in bucket[key]:
            
            if key[3] == 'n':        
                train_data_x.append(element)
                train_data_y.append("N")
            elif key[3] == 'H':
                train_data_x.append(element)
                train_data_y.append("Y")
                
#         [str_1 for ele in test_data: if ele[1] == 'n': str_1 = 'N' else: str_1 ='Y']
    
    for ele in test_data_y:
        if ele == 'n':
            test_list_y.append("N")
        else:
            test_list_y.append("Y")
    
    print(len(test_list_y)) 
    
    values = np.array(train_data_y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(np.array(integer_encoded))

    test_values = np.array(test_list_y)
    integer_test = label_encoder.transform(test_values)
    integer_test = integer_test.reshape(len(integer_test), 1)
    onehot_test = onehot_encoder.transform(np.array(integer_test))

    train_object = Data()
    # placing X and Y values in the object of class Data 
    train_object.data = train_data_x
    train_object.target = onehot_encoded
    train_object.num_examples = len(train_data_x)

    test_object = Data()
    test_object.data = test_data_x
    test_object.target = onehot_test
    test_object.num_examples = len(test_data_x)

    return [train_object,test_object]

class Data(): 
    
    def __init__(self):
    
        self.num_epochs_completed = 0 
        self.index_in_epoch = 0 
    
    def next_batch(self,batch_size):
        
        start = self.index_in_epoch
        
        self.index_in_epoch += batch_size 
        
        # For now skipping the last batch if the size of the batch is less than batch_size 
        # Later will add a placeholder for batch_size so that we can have a dynamic batch_size at runtime 
        
        if self.index_in_epoch > self.num_examples:
            
            self.num_epochs_completed += 1 
            
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            
            self.data = [self.data[i] for i in perm]
            self.target = [self.target[i] for i in perm]
            
            start = 0
            
            self.index_in_epoch = batch_size
            
            assert batch_size <= self.num_examples
            
        end = self.index_in_epoch
        
        batch_x = []
        batch_y = [] 
        
        for i in range(start,end):
            batch_x.append(self.data[i])
            batch_y.append(self.target[i])
            
#             print(len(batch_x))
        return [batch_x,batch_y]

tic = time.time() 

#Reading the dataset
curr_dir = './dataset/2'
files = os.listdir(curr_dir)
n_files = len(files)

bucket = utilities.Utilities.read_dataset_one_bucket(curr_dir,files,inc)

bucket = standardizing(bucket)
# Ethylene in high concentration and no concentration only 
train_object, test_object = train_test_scenario_one(bucket)

############### END ###############################

settings = {'hidden_units_d' : 10, 'seq_length' : 297, 'cond_dim' : 2, 'batch_size': 10, 'hidden_units_g' : 10, 'num_generated_features' : 10,'num_signals' : 10,'wrong_labels' : True, 'latent_dim' : 10, 'learning_rate' : 0.01 , 'D_rounds' : 10 , 'G_rounds' : 10 } 
iteration = 10
GANs.build_gan_model(train_object,settings,iteration,one_hot = True )

