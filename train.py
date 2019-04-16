import tensorflow as tf

import os
from began import BEGAN


import math
import numpy as np

import utils
from imageiteartor import ImageIterator


def train(data_root, model, total_epoch, batch_size, lrate):

    X, Z, Lr, Kt = model.inputs()
    d_loss, g_loss, real_loss, fake_loss = model.loss(X, Z, Kt)
    d_opt, g_opt = model.optimizer(d_loss, g_loss, Lr)
    g_sample = model.sample(Z)
    sample_size = batch_size
    test_noise = utils.get_noise(sample_size, n_noise)
    epoch_drop = 3


    _lambda = 0.001
    _gamma = 0.5
    _kt = 0.0

    iterator, image_count = ImageIterator(data_root, batch_size, model.image_size, model.image_channels).get_iterator()
    next_element = iterator.get_next()

    measure = real_loss + tf.abs(_gamma * real_loss - fake_loss)
    tf.summary.scalar('measure', measure)

    merged = tf.summary.merge_all()



    total_batch = int(image_count/batch_size)
    #learning_rate = lrate
    #G_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('./logs',
                                      sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)

        for epoch in range(total_epoch):
            learning_rate = lrate * \
                            math.pow(0.2, math.floor((epoch + 1) / epoch_drop))
            for step in range(total_batch):
                batch_x = sess.run(next_element)
                batch_z = utils.get_uniform_noise(batch_size, n_noise)

                _, val_real_loss = sess.run([d_opt, real_loss],
                                        feed_dict={X: batch_x, Z: batch_z, Lr: learning_rate, Kt: _kt})
                _, val_fake_loss = sess.run([g_opt, fake_loss],
                                        feed_dict={Z: batch_z, Lr: learning_rate, Kt: _kt})

                _kt = _kt + _lambda * (_gamma * val_real_loss - val_fake_loss)

                if step % 300 == 0:
                    summary = sess.run(merged,
                                        feed_dict={X: batch_x, Z: batch_z, Lr: learning_rate, Kt: _kt})
                    train_writer.add_summary(summary, epoch*total_batch+step)

                    val_measure = val_real_loss + np.abs(_gamma * val_real_loss - val_fake_loss)

                    
                    print('Epoch:', '%04d' % epoch,
                    '%05d/%05d' % (step, total_batch),
                    'measure: {:.4}'.format(val_measure))
                    
                    #sample_size = 10
                    #noise = get_noise(sample_size, n_noise)
                    samples = sess.run(g_sample, feed_dict={Z: test_noise})
                    title = 'samples/%05d_%05d.png'%(epoch, step)
                    utils.save_samples(title, samples)


            saver.save(sess, './models/began', global_step=epoch)






if __name__ == "__main__":
    #set hyper parameters
    batch_size = 16
    n_noise = 64
    image_size = 64
    image_channels = 3
    learning_rate = 0.0001
    total_epochs = 10

    model = BEGAN(batch_size, n_noise, image_size, image_channels)
    
    #download align_celeba dataset from https://www.kaggle.com/jessicali9530/celeba-dataset
    #extract and move to "./data/img_align_celeba"
    data_root = './data/img_align_celeba'

    with tf.Graph().as_default():
        train(data_root, model, total_epochs, batch_size, learning_rate)



