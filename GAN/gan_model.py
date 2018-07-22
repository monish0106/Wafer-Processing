import tensorflow as tf 
from tensorflow.python.ops.rnn_cell import LSTMCell
import numpy as np

def sample_Z(batch_size, seq_length, latent_dim):
    sample = np.float32(np.random.normal(size=[batch_size, seq_length, latent_dim]))
    return sample

def sample_C(batch_size, cond_dim=0, one_hot=False):
    
    if cond_dim == 0:
        return None
    else:

        C = np.zeros(shape=(batch_size, cond_dim))

        if one_hot == True:
            labels = np.random.choice(cond_dim, batch_size)
            C[np.arange(batch_size), labels] = 1
        else:
            C = np.float32(np.random.normal(size=[batch_size, cond_dim]))
        return C 
            
def create_placeholders(batch_size, seq_length, latent_dim, num_generated_features, cond_dim):
    Z = tf.placeholder(tf.float32, [batch_size, seq_length, latent_dim])
    X = tf.placeholder(tf.float32, [batch_size, seq_length, num_generated_features])
    CG = tf.placeholder(tf.float32, [batch_size, cond_dim])
    CD = tf.placeholder(tf.float32, [batch_size, cond_dim])
    CS = tf.placeholder(tf.float32, [batch_size, cond_dim])
    return Z, X, CG, CD, CS

def generator(z, hidden_units_g, seq_length, batch_size, num_generated_features, reuse=False, parameters=None, cond_dim=0, c=None, learn_scale=True):
    """
    If parameters are supplied, initialise as such
    """
    with tf.variable_scope("generator") as scope:
        if reuse:
            scope.reuse_variables()
        if parameters is None:
            W_out_G_initializer = tf.truncated_normal_initializer()
            b_out_G_initializer = tf.truncated_normal_initializer()
            scale_out_G_initializer = tf.constant_initializer(value=1.0)
            lstm_initializer = None
            bias_start = 1.0
        else:
            W_out_G_initializer = tf.constant_initializer(value=parameters['generator/W_out_G:0'])
            b_out_G_initializer = tf.constant_initializer(value=parameters['generator/b_out_G:0'])
            try:
                scale_out_G_initializer = tf.constant_initializer(value=parameters['generator/scale_out_G:0'])
            except KeyError:
                scale_out_G_initializer = tf.constant_initializer(value=1)
                assert learn_scale
            lstm_initializer = tf.constant_initializer(value=parameters['generator/rnn/lstm_cell/weights:0'])
            bias_start = parameters['generator/rnn/lstm_cell/biases:0']

        W_out_G = tf.get_variable(name='W_out_G', shape=[hidden_units_g, num_generated_features], initializer=W_out_G_initializer)
        b_out_G = tf.get_variable(name='b_out_G', shape=num_generated_features, initializer=b_out_G_initializer)
        scale_out_G = tf.get_variable(name='scale_out_G', shape=1, initializer=scale_out_G_initializer, trainable=learn_scale)
        if cond_dim > 0:
            # CGAN!
            assert not c is None
            repeated_encoding = tf.stack([c]*seq_length, axis=1)
            inputs = tf.concat([z, repeated_encoding], axis=2)

            #repeated_encoding = tf.tile(c, [1, tf.shape(z)[1]])
            #repeated_encoding = tf.reshape(repeated_encoding, [tf.shape(z)[0], tf.shape(z)[1], cond_dim])
            #inputs = tf.concat([repeated_encoding, z], 2)
        else:
            inputs = z

        cell = LSTMCell(num_units=hidden_units_g,
                           state_is_tuple=True,
                           initializer=lstm_initializer,
                           reuse=reuse)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            sequence_length=[seq_length]*batch_size,
            inputs=inputs)
        rnn_outputs_2d = tf.reshape(rnn_outputs, [-1, hidden_units_g])
        logits_2d = tf.matmul(rnn_outputs_2d, W_out_G) + b_out_G
#        output_2d = tf.multiply(tf.nn.tanh(logits_2d), scale_out_G)
        output_2d = tf.nn.tanh(logits_2d)
        output_3d = tf.reshape(output_2d, [-1, seq_length, num_generated_features])
    return output_3d

def discriminator(x, hidden_units_d, seq_length, batch_size, reuse=False, 
        cond_dim=0, c=None, batch_mean=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        W_out_D = tf.get_variable(name='W_out_D', shape=[hidden_units_d, 1],
                initializer=tf.truncated_normal_initializer())
        b_out_D = tf.get_variable(name='b_out_D', shape=1,
                initializer=tf.truncated_normal_initializer())
#        W_final_D = tf.get_variable(name='W_final_D', shape=[hidden_units_d, 1],
#                initializer=tf.truncated_normal_initializer())
#        b_final_D = tf.get_variable(name='b_final_D', shape=1,
#                initializer=tf.truncated_normal_initializer())

        if cond_dim > 0:
            assert not c is None
            repeated_encoding = tf.stack([c]*seq_length, axis=1)
            inputs = tf.concat([x, repeated_encoding], axis=2)
        else:
            inputs = x
        # add the average of the inputs to the inputs (mode collapse?
        if batch_mean:
            mean_over_batch = tf.stack([tf.reduce_mean(x, axis=0)]*batch_size, axis=0)
            inputs = tf.concat([x, mean_over_batch], axis=2)

        cell = tf.contrib.rnn.LSTMCell(num_units=hidden_units_d, 
                state_is_tuple=True,
                reuse=reuse)
        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
            cell=cell,
            dtype=tf.float32,
            inputs=inputs)
#        logit_final = tf.matmul(rnn_outputs[:, -1], W_final_D) + b_final_D
        logits = tf.einsum('ijk,km', rnn_outputs, W_out_D) + b_out_D
#        rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, hidden_units_d])
#        logits = tf.matmul(rnn_outputs_flat, W_out_D) + b_out_D
        output = tf.nn.sigmoid(logits)
    #return output, logits, logit_final
    return output, logits


def GAN_loss(Z, X, generator_settings, discriminator_settings, cond, CG, CD, CS, wrong_labels=False):
    
    if cond:
        G_sample = generator(Z, **generator_settings, c=CG)
        D_real, D_logit_real =  discriminator(X, **discriminator_settings, c=CD)
        D_fake, D_logit_fake = discriminator(G_sample, reuse=True, **discriminator_settings, c=CG)
        
        if wrong_labels:
            D_wrong, D_logit_wrong = discriminator(X, reuse=True, **discriminator_settings, c=CS)
    else:
        G_sample = generator(Z, **generator_settings)
        D_real, D_logit_real  = discriminator(X, **discriminator_settings)
        D_fake, D_logit_fake = discriminator(G_sample, reuse=True, **discriminator_settings)

    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)), 1)
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)), 1)
    D_loss_wrong = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_wrong, labels=tf.zeros_like(D_logit_wrong)), 1)
   

    D_loss = D_loss_real + D_loss_fake

    if cond and wrong_labels:
        D_loss = D_loss + D_loss_wrong

    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)), 1)
    
    return D_loss, G_loss

def GAN_solvers(D_loss, G_loss, learning_rate, batch_size):

    discriminator_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
    generator_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
    
    D_loss_mean_over_batch = tf.reduce_mean(D_loss)
    D_solver = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(D_loss_mean_over_batch, var_list=discriminator_vars)
    G_loss_mean_over_batch = tf.reduce_mean(G_loss)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss_mean_over_batch, var_list=generator_vars)

    return D_solver, G_solver



                                                   
                                                 

