import gan_model as model 
import tensorflow as tf 
import numpy as np 

def build_gan_model(train_object,settings,iteration,one_hot = True ):
    
    batch_size = settings['batch_size']
    seq_length = settings['seq_length']
    latent_dim = settings['latent_dim']
    num_signals = settings['num_signals']
    cond_dim = settings['cond_dim']
    wrong_labels = settings['wrong_labels']
    learning_rate = settings['learning_rate']
    D_rounds = settings['D_rounds']
    G_rounds = settings['G_rounds']
    
    Z, X, CG, CD, CS = model.create_placeholders(batch_size, seq_length, latent_dim, 
                                    num_signals, cond_dim)
    
    discriminator_vars = ['hidden_units_d', 'seq_length', 'cond_dim', 'batch_size']
    discriminator_settings = dict((k, settings[k]) for k in discriminator_vars)
    generator_vars = ['hidden_units_g', 'seq_length', 'batch_size', 'num_generated_features', 'cond_dim']
    generator_settings = dict((k, settings[k]) for k in generator_vars)
    
    
    
    CGAN = (cond_dim > 0)
    
    D_loss, G_loss = model.GAN_loss(Z, X, generator_settings, discriminator_settings, CGAN, CG, CD, CS, wrong_labels= wrong_labels)
    D_solver, G_solver = model.GAN_solvers(D_loss, G_loss, learning_rate, batch_size)
    
    G_sample = model.generator(Z, **generator_settings, reuse=True, c=CG)
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())

        for i in range(iteration):
            print("ITeration number: " ,i)
            for d in range(D_rounds):
                
                X_mb, Y_mb = train_object.next_batch(batch_size)
                
                Z_mb = model.sample_Z(batch_size, seq_length, latent_dim)
                if cond_dim > 0:
                    # CGAN
                    Y_mb = np.asarray(Y_mb)
                    Y_mb = Y_mb.reshape(-1, cond_dim)
                    if one_hot:
                        # change all of the labels to a different one
                        offsets = np.random.choice(cond_dim-1, batch_size) + 1
                        new_labels = (np.argmax(Y_mb, axis=1) + offsets) % cond_dim
                        Y_wrong = np.zeros_like(Y_mb)
                        Y_wrong[np.arange(batch_size), new_labels] = 1
                    else:
                        # flip all of the bits (assuming binary...)
                        Y_wrong = 1 - Y_mb
                    _ = sess.run(D_solver, feed_dict={X: X_mb, Z: Z_mb, CD: Y_mb, CS: Y_wrong, CG: Y_mb})
                else:
                    _ = sess.run(D_solver, feed_dict={X: X_mb, Z: Z_mb})
             # update the generator
            for g in range(G_rounds):
                if cond_dim > 0:
                    # note we are essentially throwing these X_mb away...
                    X_mb, Y_mb = train_object.next_batch(batch_size)
                    _ = sess.run(G_solver,
                            feed_dict={Z: model.sample_Z(batch_size, seq_length, latent_dim), CG: Y_mb})
                else:
                    _ = sess.run(G_solver,
                            feed_dict={Z: model.sample_Z(batch_size, seq_length, latent_dim)})
        # at the end, get the loss
        if cond_dim > 0:
            D_loss_curr, G_loss_curr = sess.run([D_loss, G_loss], feed_dict={X: X_mb, Z: model.sample_Z(batch_size, seq_length, latent_dim), CG: Y_mb, CD: Y_mb})
            D_loss_curr = np.mean(D_loss_curr)
            G_loss_curr = np.mean(G_loss_curr)
        else:
            D_loss_curr, G_loss_curr = sess.run([D_loss, G_loss], feed_dict={X: X_mb, Z: model.sample_Z(batch_size, seq_length, latent_dim)})
            D_loss_curr = np.mean(D_loss_curr)
            G_loss_curr = np.mean(G_loss_curr)

    return D_loss_curr, G_loss_curr
    
    
    
    
    
    

