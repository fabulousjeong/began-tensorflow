import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.initializers import TruncatedNormal


class BEGAN(object):
    def __init__(self):
        self.n_W1 = 64

    # graw feature map using conv2d_transpose
    # use batch_norm and relu
    # output size is 64x64

    def decoder(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(8*8*self.n_W1, use_bias=True, input_shape=(100,), kernel_initializer = RandomNormal(mean=0.0, stddev=0.01, seed=None)))
        model.add(layers.Reshape((8, 8, self.n_W1)))
        assert model.output_shape == (None, 8, 8, self.n_W1)

        model.add(layers.Conv2D(self.n_W1, (3, 3), strides=(1, 1), padding='same', activation='elu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02, seed=None)))
        model.add(layers.Conv2D(self.n_W1, (3, 3), strides=(1, 1), padding='same', activation='elu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02, seed=None)))
        model.add(tf.keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest'))
        assert model.output_shape == (None, 16, 16, self.n_W1)

        
        model.add(layers.Conv2D(self.n_W1, (3, 3), strides=(1, 1), padding='same', activation='elu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02, seed=None)))
        model.add(layers.Conv2D(self.n_W1, (3, 3), strides=(1, 1), padding='same', activation='elu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02, seed=None)))
        model.add(tf.keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest'))
        assert model.output_shape == (None, 32, 32, self.n_W1)

        model.add(layers.Conv2D(self.n_W1, (3, 3), strides=(1, 1), padding='same', activation='elu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02, seed=None)))
        model.add(layers.Conv2D(self.n_W1, (3, 3), strides=(1, 1), padding='same', activation='elu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02, seed=None)))
        model.add(tf.keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest'))
        assert model.output_shape == (None, 64, 64, self.n_W1)

        model.add(layers.Conv2D(3, (3, 3), strides=(1, 1), padding='same', activation='sigmoid', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02, seed=None)))
        assert model.output_shape == (None, 64, 64, 3)

        return model


    def encoder(self):

        model = tf.keras.Sequential()
        model.add(layers.Conv2D(self.n_W1, (3, 3), strides=(1, 1), input_shape=[64, 64, 3], padding='same', activation='elu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02, seed=None)))
        assert model.output_shape == (None, 64, 64, self.n_W1)
        model.add(layers.Conv2D(self.n_W1, (3, 3), strides=(1, 1), padding='same', activation='elu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02, seed=None)))
        model.add(layers.Conv2D(2*self.n_W1, (3, 3), strides=(1, 1), padding='same', activation='elu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02, seed=None)))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
        assert model.output_shape == (None, 32, 32, 2*self.n_W1)

        model.add(layers.Conv2D(2*self.n_W1, (3, 3), strides=(1, 1), padding='same', activation='elu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02, seed=None)))
        model.add(layers.Conv2D(3*self.n_W1, (3, 3), strides=(1, 1), padding='same', activation='elu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02, seed=None)))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
        assert model.output_shape == (None, 16, 16, 3*self.n_W1)

        model.add(layers.Conv2D(3*self.n_W1, (3, 3), strides=(1, 1), padding='same', activation='elu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02, seed=None)))
        model.add(layers.Conv2D(4*self.n_W1, (3, 3), strides=(1, 1), padding='same', activation='elu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02, seed=None)))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
        assert model.output_shape == (None, 8, 8, 4*self.n_W1)

        model.add(layers.Conv2D(4*self.n_W1, (3, 3), strides=(1, 1), padding='same', activation='elu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02, seed=None)))
        model.add(layers.Conv2D(4*self.n_W1, (3, 3), strides=(1, 1), padding='same', activation='elu', kernel_initializer=TruncatedNormal(mean=0.0, stddev=0.02, seed=None)))

        model.add(layers.Flatten())
        assert model.output_shape == (None, 8*8*4*self.n_W1)

        model.add(layers.Dense(100, use_bias=True, activation='tanh', kernel_initializer = RandomNormal(mean=0.0, stddev=0.01, seed=None)))

        return model


    def generator(self):
        inputs = tf.keras.Input(shape=(100,))
        outputs = self.decoder()(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model

    def discriminator(self):
        inputs = tf.keras.Input(shape=(64, 64, 3))
        x = self.encoder()(inputs)
        outputs = self.decoder()(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    # Loss function and optimizer are same as simple GAN
    def loss(self, X, g_out, d_real, d_fake, Kt):

        real_loss = tf.reduce_mean(tf.abs(X - d_real))
        fake_loss = tf.reduce_mean(tf.abs(g_out - d_fake))
        d_loss = real_loss - Kt * fake_loss
        g_loss = fake_loss

        return d_loss, g_loss, real_loss, fake_loss

    def optimizer(self, learning_rate):
        d_opt = tf.keras.optimizers.Adam(learning_rate, beta_1 = 0.5, beta_2=0.999)
        g_opt = tf.keras.optimizers.Adam(learning_rate, beta_1 = 0.5, beta_2=0.999)
        return d_opt, g_opt


