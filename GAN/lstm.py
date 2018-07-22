import tensorflow as tf 
from tensorflow.contrib import rnn
import numpy as np 

def basic_lstm_technique(data,test_data,time_steps = 400, num_units = 32, n_input = 10, lr= 0.001,n_classes = 2,bs = 4):

    #weights and biases of appropriate shape to accomplish above task
    out_weights=tf.Variable(tf.random_normal([num_units,n_classes]))
    out_bias=tf.Variable(tf.random_normal([n_classes]))
    
    #defining placeholders
    #input placeholder
    x=tf.placeholder("float",[None,time_steps,n_input])
    #target label placeholder
    y=tf.placeholder("float",[None,n_classes])

    learning_rate = tf.placeholder("float")
    
    batch_size = tf.placeholder_with_default(bs,shape= None)
    #processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
    input=tf.unstack(x ,time_steps,1)

    #defining the network
    lstm_layer=rnn.BasicLSTMCell(num_units,forget_bias=1)
    outputs,_=rnn.static_rnn(lstm_layer,input,dtype="float32")
    
    #converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
    prediction=tf.matmul(outputs[-1],out_weights)+out_bias

    #loss_function
    ratio = 114.0 / (114.0 + 30.0)
    class_weight = tf.constant([ ratio,1.0 -  ratio])
    weighted_prediction = tf.multiply(prediction, class_weight)
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=weighted_prediction,labels=y))
    #optimization
    opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    #model evaluation
    correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    #initialize variables
    init=tf.global_variables_initializer()
    with tf.Session() as sess:
       
        sess.run(init)
        
        iter=1
        while iter< 50:

   
            batch_x,batch_y= data.next_batch(bs)
            print((batch_x).shape)
            batch_x = np.asarray(batch_x)
#             lr = lr / (iter +1) 
                   
            sess.run(opt, feed_dict={x: batch_x, y: batch_y,learning_rate: lr,batch_size: bs})
        
            if iter /1==iter:
                
                acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
                los=sess.run(loss,feed_dict={x:batch_x,y:batch_y})
                print("---------------------------------")
                print("For iter ",iter)
                print("Accuracy ",acc)
                print("Loss ",los)
                print("Epochs : ",data.num_epochs_completed)
                print("Steps : " ,data.index_in_epoch)
                print("---------------------------------")

            iter=iter+1
        batch_x,batch_y= data.next_batch(data.num_examples)
        print(len(batch_y))
        batch_x = np.asarray(batch_x)
        print("Training Accuracy:", sess.run(accuracy, feed_dict={x: batch_x, y: batch_y,batch_size : (data.num_examples)}))   
        
        batch_x,batch_y= test_data.next_batch(test_data.num_examples)
        print(len(batch_y))
        batch_x = np.asarray(batch_x)
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: batch_x, y: batch_y,batch_size : (test_data.num_examples)}))
        