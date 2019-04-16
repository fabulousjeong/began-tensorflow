import tensorflow as tf

import os
from began import BEGAN

import math

import utils
from imageiteartor import ImageIterator
import datetime
import numpy as np

def test(ckpt_root, model, batch_size):

    X, Z, Lr, Kt = model.inputs()
    g_sample = model.sample(Z, reuse = False)
    sample_size = batch_size
    test_noise = utils.get_uniform_noise(sample_size, n_noise)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt_root)

        samples = sess.run(g_sample, feed_dict={Z: test_noise})
        date = datetime.datetime.now()
        title = 'samples/%s.png' % date
        utils.save_samples(title, samples)
    np.savetxt('samples/test_noise%s.txt' % date, test_noise, fmt='%2.5f', delimiter=', ')


    


if __name__ == "__main__":
    #set hyper parameters
    batch_size = 36
    n_noise = 64
    image_size = 64
    image_channels = 3

    tf.reset_default_graph()

    model = BEGAN(batch_size, n_noise, image_size, image_channels)

    ckpt_root = tf.train.latest_checkpoint('models',latest_filename=None)
#'./models/began-16x16-9'

    with tf.Graph().as_default():
        test(ckpt_root, model, batch_size)



