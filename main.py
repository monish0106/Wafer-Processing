# Importing the required packages
from collections import Counter
from code import utilities
import os 
import random 
import lstm 
import numpy as np 
import time 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

import logging
LOG_FILENAME = 'example.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)

num_examples = 400 

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
            batch_x.append(self.data[i][0])
            batch_y.append(self.target[i])
        
        return [batch_x,batch_y]



tic = time.time() 

#Reading the dataset
curr_dir = './dataset/2'
files = os.listdir(curr_dir)
print(files)
n_files = len(files)

# logging.debug('Reading the files : ',n_files,'\nTime :',time.time() - tic) 

data = [] 
for i in range(n_files):
    data.append(utilities.Utilities.read_dataset_one(curr_dir,files[i],num_examples))
    
# Train Test split 
random.shuffle(data)

n_train = int(0.8 * n_files) 

train_data = data[0:n_train]
test_data = data[n_train:]

train_object = Data()

# Changing all Ys to one hot vectors 
all_y = [] 
for i in range(len(train_data)):
    all_y.append(train_data[i][1])
    
values = np.array(all_y)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(np.array(integer_encoded))

test_y = [] 
for i in range(len(test_data)):
    test_y.append(test_data[i][1])

test_values = np.array(test_y)
integer_test = label_encoder.transform(test_values)
integer_test = integer_test.reshape(len(integer_test), 1)
onehot_test = onehot_encoder.transform(np.array(integer_test))

# placing X and Y values in the object of class Data 
train_object.data = train_data
train_object.target = onehot_encoded
train_object.num_examples = len(train_data)

print(train_object.num_examples)
print(onehot_encoded)
test_object = Data()
test_object.data = test_data
test_object.target = onehot_test
test_object.num_examples = len(test_data)

# plot() 
print(train_object.data)

#Classification 
lstm.basic_lstm_technique(train_object,test_object,time_steps = num_examples, num_units = 32, n_input = 10, lr = 0.01,n_classes = 2,bs = train_object.num_examples)


