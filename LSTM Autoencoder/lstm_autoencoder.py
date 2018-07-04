# Basic libraries
import numpy as np
import tensorflow as tf

from LSTMAutoencoder import *

def basic_lstm_autoencoder(train_object,test_object,bs = 128, num_hidden = 12, time_steps = 8, n_inputs = 1, iteration = 10000,lr = 0.01):
    
    #placeholder list
    p_input = tf.placeholder(tf.float32, shape=(bs, time_steps, n_inputs))

    p_inputs = [tf.squeeze(t, [1]) for t in tf.split(p_input, time_steps, 1)]
    cell = tf.nn.rnn_cell.LSTMCell(num_hidden, use_peepholes=True)
    
    ae = LSTMAutoencoder(num_hidden, p_inputs,lr, cell=cell)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

   
        for i in range(iteration):
            
            batch_x,_ = train_object.next_batch(bs)
            batch_x = np.asarray(batch_x)
   
            (loss_val, _) = sess.run([ae.loss, ae.train], {p_input: batch_x})
            print('iter %d:' % (i + 1), loss_val)

        (input_, output_,loss_) = sess.run([ae.input_, ae.output_,ae.loss], {p_input: train_object.data})
        print('train result :',loss_)
        print('input :', input_[0, :, :].flatten())
        print('output :', output_[0, :, :].flatten())


    
    
    


    
    
    
    
    
    