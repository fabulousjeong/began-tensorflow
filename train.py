import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)
from began import BEGAN
import utils

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
import time

import glob
import imageio
import math



def train(train_dir, model, total_epoch, batch_size, lrate):

    train_image_generator = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
    train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                            directory=train_dir,
                                                            shuffle=True,
                                                            target_size=(64, 64))

    print(f'num: {len(train_data_gen)}')

    generator = model.generator()
    discriminator = model.discriminator()

    discriminator_optimizer, generator_optimizer = model.optimizer(lrate)

    noise_dim = 100
    num_examples_to_generate = 16
    epoch_drop = 50

    _lambda = 0.001
    _gamma = 0.5
    _kt = 0.0
    measure = 0


    #checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


    # We will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(images, discriminator_optimizer, generator_optimizer, _kt):
        noise = tf.random.normal([batch_size, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            g_out = generator(noise, training=True)
            d_real = discriminator(images, training=True)
            d_fake = discriminator(g_out, training=True)


            disc_loss, gen_loss, real_loss, fake_loss = model.loss(images, g_out, d_real, d_fake, _kt)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
            
        return real_loss, fake_loss


    print ('Start Training!!!')

    for epoch in range(total_epoch):
        start = time.time()
        learning_rate = lrate * \
                math.pow(0.2, math.floor((epoch + 1) / epoch_drop))
        discriminator_optimizer, generator_optimizer = model.optimizer(learning_rate)

        #len(train_data_gen)
        for i in range(len(train_data_gen)):#
            sample_training_images, _ = next(train_data_gen)
            real_loss, fake_loss = train_step(sample_training_images, discriminator_optimizer, generator_optimizer, _kt)
            _kt = _kt + _lambda * (_gamma * real_loss - fake_loss)
            measure = real_loss + np.abs(_gamma * real_loss - fake_loss)
            print(f'Training... {i + 1:04} / {len(train_data_gen):04} , measure: {measure:.5f}, _kt: {_kt:.5f} \r',end='')

        # Produce images for the GIF as we go
        utils.generate_and_save_images(generator,
                                epoch + 1,
                                seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_dir = './training_checkpoints'
            checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
            checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)
            checkpoint.save(file_prefix = checkpoint_prefix)

        print (f'\n Time for epoch {epoch + 1} is {time.time()-start} sec')

    anim_file = 'began.gif'
    utils.generate_gif(anim_file)


if __name__ == "__main__":
    #set hyper parameters
    batch_size = 64
    n_noise = 100
    learning_rate = 1e-4
    total_epochs = 100

    #model = SIMPLEGAN(batch_size, n_noise, image_size, image_channels)
    model = BEGAN()
    #data_root = '../data/mnist/trainingSet' 
    
    #download align_celeba dataset from https://www.kaggle.com/jessicali9530/celeba-dataset
    #extract and move to "./data/img_align_celeba"
    train_dir = './data/img_align_celeba'
    anim_file = 'began.gif'

    train(train_dir, model, total_epochs, batch_size, learning_rate)




