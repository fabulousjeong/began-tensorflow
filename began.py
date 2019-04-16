import tensorflow as tf



class BEGAN(object):
    def __init__(self, batch_size, n_noise, image_size, image_channels):
        self.batch_size = batch_size
        self.n_noise = n_noise

        # but we only take 64x64
        self.image_size = image_size
        self.image_channels = image_channels
        self.n_input = image_size*image_size*image_channels
        self.n_W1 = 64

        self.n_hidden = 8*8*self.n_W1

    # this model take input image and noise vector Z
    # can adjust learning_rate using Lr parameter
    # can adjust learning_rate using Lr parameter
    def inputs(self):
        X = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.image_channels], name='input_sample')
        Z = tf.placeholder(tf.float32, [None, self.n_noise], name='input_noise')
        Lr = tf.placeholder(tf.float32, [], name='learning_rate')
        Kt = tf.placeholder(tf.float32, [], name='equilibrium_rate')
        return X, Z, Lr, Kt
    

    # graw feature map using conv2d_transpose
    # use batch_norm and relu
    # output size is 64x64

    def decoder(self, input):

        D_FW1 = tf.get_variable('D_FW1', [self.n_noise, self.n_hidden], initializer = tf.random_normal_initializer(stddev=0.01))
        D_Fb1 = tf.get_variable('D_Fb1', [self.n_hidden], initializer = tf.constant_initializer(0))
        D_W1 = tf.get_variable('D_W1', [3,3,self.n_W1, self.n_W1], initializer = tf.truncated_normal_initializer(stddev=0.02))
        D_b1 = tf.get_variable('D_b1', [self.n_W1], initializer = tf.constant_initializer(0))     

        D_W2 = tf.get_variable('D_W2', [3,3,self.n_W1, 2*self.n_W1], initializer = tf.truncated_normal_initializer(stddev=0.02))
        D_b2 = tf.get_variable('D_b2', [2*self.n_W1], initializer = tf.constant_initializer(0))     

        D_W3 = tf.get_variable('D_W3', [3,3,2*self.n_W1, self.n_W1], initializer = tf.truncated_normal_initializer(stddev=0.02))
        D_b3 = tf.get_variable('D_b3', [self.n_W1], initializer = tf.constant_initializer(0))     

        D_W4 = tf.get_variable('D_W4', [3,3,self.n_W1, 2*self.n_W1], initializer = tf.truncated_normal_initializer(stddev=0.02))
        D_b4 = tf.get_variable('D_b4', [2*self.n_W1], initializer = tf.constant_initializer(0))     

        D_W5 = tf.get_variable('D_W5', [3,3,2*self.n_W1, self.n_W1], initializer = tf.truncated_normal_initializer(stddev=0.02))
        D_b5 = tf.get_variable('D_b5', [self.n_W1], initializer = tf.constant_initializer(0))     

        D_W6 = tf.get_variable('D_W6', [3,3,self.n_W1, 2*self.n_W1], initializer = tf.truncated_normal_initializer(stddev=0.02))
        D_b6 = tf.get_variable('D_b6', [2*self.n_W1], initializer = tf.constant_initializer(0))     

        D_W7 = tf.get_variable('D_W7', [3,3,2*self.n_W1, self.n_W1], initializer = tf.truncated_normal_initializer(stddev=0.02))
        D_b7 = tf.get_variable('D_b7', [self.n_W1], initializer = tf.constant_initializer(0))     

        D_W8 = tf.get_variable('D_W8', [3,3,self.n_W1, self.n_W1], initializer = tf.truncated_normal_initializer(stddev=0.02))
        D_b8 = tf.get_variable('D_b8', [self.n_W1], initializer = tf.constant_initializer(0))    

        D_W9 = tf.get_variable('D_W9', [3,3,self.n_W1, self.image_channels], initializer = tf.truncated_normal_initializer(stddev=0.02))
        D_b9 = tf.get_variable('D_b9', [self.image_channels], initializer = tf.constant_initializer(0))     
        

        hidden = tf.nn.relu(
                tf.matmul(input, D_FW1) + D_Fb1)
        hidden = tf.reshape(hidden, [self.batch_size, 8,8,self.n_W1]) 

        conv1 = tf.nn.conv2d(hidden, D_W1, strides = [1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.bias_add(conv1, D_b1)
        conv1 = tf.nn.elu(conv1)
        conv2 = tf.nn.conv2d(conv1, D_W2, strides = [1, 1, 1, 1], padding='SAME')
        conv2 = tf.nn.bias_add(conv2, D_b2)
        conv2 = tf.nn.elu(conv2)
        conv2 = tf.image.resize_nearest_neighbor(conv2, (16,16))

        conv3 = tf.nn.conv2d(conv2, D_W3, strides = [1, 1, 1, 1], padding='SAME')
        conv3 = tf.nn.bias_add(conv3, D_b3)
        conv3 = tf.nn.elu(conv3)
        conv4 = tf.nn.conv2d(conv3, D_W4, strides = [1, 1, 1, 1], padding='SAME')
        conv4 = tf.nn.bias_add(conv4, D_b4)
        conv4 = tf.nn.elu(conv4)
        conv4 = tf.image.resize_nearest_neighbor(conv4, (32,32))

        conv5 = tf.nn.conv2d(conv4, D_W5, strides = [1, 1, 1, 1], padding='SAME')
        conv5 = tf.nn.bias_add(conv5, D_b5)
        conv5 = tf.nn.elu(conv5)
        conv6 = tf.nn.conv2d(conv5, D_W6, strides = [1, 1, 1, 1], padding='SAME')
        conv6 = tf.nn.bias_add(conv6, D_b6)
        conv6 = tf.nn.elu(conv6)
        conv6 = tf.image.resize_nearest_neighbor(conv6, (64,64))

        conv7 = tf.nn.conv2d(conv6, D_W7, strides = [1, 1, 1, 1], padding='SAME')
        conv7 = tf.nn.bias_add(conv7, D_b7)
        conv7 = tf.nn.elu(conv7)
        conv8 = tf.nn.conv2d(conv7, D_W8, strides = [1, 1, 1, 1], padding='SAME')
        conv8 = tf.nn.bias_add(conv8, D_b8)
        conv8 = tf.nn.elu(conv8)
        conv9 = tf.nn.conv2d(conv8, D_W9, strides = [1, 1, 1, 1], padding='SAME')
        conv9 = tf.nn.bias_add(conv9, D_b9)
        output = tf.nn.tanh(conv9)
        return output


    def encoder(self, input):

        E_W0 = tf.get_variable('E_W0', [3,3,self.image_channels, self.n_W1], initializer = tf.truncated_normal_initializer(stddev=0.02))
        E_b0 = tf.get_variable('E_b0', [self.n_W1], initializer = tf.constant_initializer(0))     

        E_W1 = tf.get_variable('E_W1', [3,3,self.n_W1, self.n_W1], initializer = tf.truncated_normal_initializer(stddev=0.02))
        E_b1 = tf.get_variable('E_b1', [self.n_W1], initializer = tf.constant_initializer(0))     

        E_W2 = tf.get_variable('E_W2', [3,3,self.n_W1, 2*self.n_W1], initializer = tf.truncated_normal_initializer(stddev=0.02))
        E_b2 = tf.get_variable('E_b2', [2*self.n_W1], initializer = tf.constant_initializer(0))     

        E_W3 = tf.get_variable('E_W3', [3,3,2*self.n_W1, 2*self.n_W1], initializer = tf.truncated_normal_initializer(stddev=0.02))
        E_b3 = tf.get_variable('E_b3', [2*self.n_W1], initializer = tf.constant_initializer(0))     

        E_W4 = tf.get_variable('E_W4', [3,3,2*self.n_W1, 3*self.n_W1], initializer = tf.truncated_normal_initializer(stddev=0.02))
        E_b4 = tf.get_variable('E_b4', [3*self.n_W1], initializer = tf.constant_initializer(0))     

        E_W5 = tf.get_variable('E_W5', [3,3,3*self.n_W1, 3*self.n_W1], initializer = tf.truncated_normal_initializer(stddev=0.02))
        E_b5 = tf.get_variable('E_b5', [3*self.n_W1], initializer = tf.constant_initializer(0))     

        E_W6 = tf.get_variable('E_W6', [3,3,3*self.n_W1, 3*self.n_W1], initializer = tf.truncated_normal_initializer(stddev=0.02))
        E_b6 = tf.get_variable('E_b6', [3*self.n_W1], initializer = tf.constant_initializer(0))     

        E_W7 = tf.get_variable('E_W7', [3,3,3*self.n_W1, 3*self.n_W1], initializer = tf.truncated_normal_initializer(stddev=0.02))
        E_b7 = tf.get_variable('E_b7', [3*self.n_W1], initializer = tf.constant_initializer(0))     

        E_W8 = tf.get_variable('E_W8', [3,3,3*self.n_W1, 3*self.n_W1], initializer = tf.truncated_normal_initializer(stddev=0.02))
        E_b8 = tf.get_variable('E_b8', [3*self.n_W1], initializer = tf.constant_initializer(0))  

        E_FW1 = tf.get_variable('E_FW1', [3*self.n_hidden, self.n_noise], initializer = tf.random_normal_initializer(stddev=0.01))
        E_Fb1 = tf.get_variable('E_Fb1', [self.n_noise], initializer = tf.constant_initializer(0))     
    
        conv0 = tf.nn.conv2d(input, E_W0, strides = [1, 1, 1, 1], padding='SAME')
        conv0 = tf.nn.bias_add(conv0, E_b0)
        conv0 = tf.nn.elu(conv0)
        conv1 = tf.nn.conv2d(conv0, E_W1, strides = [1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.bias_add(conv1, E_b1)
        conv1 = tf.nn.elu(conv1)
        conv2 = tf.nn.conv2d(conv1, E_W2, strides = [1, 1, 1, 1], padding='SAME')
        conv2 = tf.nn.bias_add(conv2, E_b2)
        conv2 = tf.nn.elu(conv2)
        conv2 = tf.nn.max_pool(conv2, ksize = (1,2,2,1), strides = (1,2,2,1), padding='SAME')

        conv3 = tf.nn.conv2d(conv2, E_W3, strides = [1, 1, 1, 1], padding='SAME')
        conv3 = tf.nn.bias_add(conv3, E_b3)
        conv3 = tf.nn.elu(conv3)
        conv4 = tf.nn.conv2d(conv3, E_W4, strides = [1, 1, 1, 1], padding='SAME')
        conv4 = tf.nn.bias_add(conv4, E_b4)
        conv4 = tf.nn.elu(conv4)
        conv4 = tf.nn.max_pool(conv4, ksize = (1,2,2,1), strides = (1,2,2,1), padding='SAME')

        conv5 = tf.nn.conv2d(conv4, E_W5, strides = [1, 1, 1, 1], padding='SAME')
        conv5 = tf.nn.bias_add(conv5, E_b5)
        conv5 = tf.nn.elu(conv5)
        conv6 = tf.nn.conv2d(conv5, E_W6, strides = [1, 1, 1, 1], padding='SAME')
        conv6 = tf.nn.bias_add(conv6, E_b6)
        conv6 = tf.nn.elu(conv6)
        conv6 = tf.nn.max_pool(conv6, ksize = (1,2,2,1), strides = (1,2,2,1), padding='SAME')

        conv7 = tf.nn.conv2d(conv6, E_W7, strides = [1, 1, 1, 1], padding='SAME')
        conv7 = tf.nn.bias_add(conv7, E_b7)
        conv7 = tf.nn.elu(conv7)
        conv8 = tf.nn.conv2d(conv7, E_W8, strides = [1, 1, 1, 1], padding='SAME')
        conv8 = tf.nn.bias_add(conv8, E_b8)
        conv8 = tf.nn.elu(conv8)

        hidden = tf.reshape(conv8, [self.batch_size, 3*self.n_hidden]) 
        output = tf.matmul(hidden, E_FW1) + E_Fb1
        output = tf.nn.tanh(output)
        return output


    def generator(self, z, reuse = False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            x = self.decoder(z)
        return x

    # symmetrical structure with the generator
    # use conv2d with stride size 2. 
    # Same as in generator, batchnorm is placed at the end of each layer. 
    # But leaky relu is used as the activate function.
    def discriminator(self, input, reuse = False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            x = self.encoder(input)
            x = self.decoder(x)
        return x
    
    # Loss function and optimizer are same as simple GAN
    def loss(self, X, Z, Kt):

        g_out = self.generator(Z)
        d_real = self.discriminator(X)
        d_fake = self.discriminator(g_out, reuse=True)

        real_loss = tf.reduce_mean(tf.abs(X - d_real))
        fake_loss = tf.reduce_mean(tf.abs(g_out - d_fake))
        d_loss = real_loss - Kt * fake_loss
        g_loss = fake_loss

        return d_loss, g_loss, real_loss, fake_loss

    def optimizer(self, d_loss, g_loss, learning_rate):
        d_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
        g_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
        #print('G_var_list:', len(G_var_list))
        #print('D_var_list:', len(D_var_list))

        d_opt = tf.train.AdamOptimizer(learning_rate, beta1 = 0.5, beta2=0.999).minimize(d_loss,
                                                                var_list=d_var_list)
        g_opt = tf.train.AdamOptimizer(learning_rate, beta1 = 0.5, beta2=0.999).minimize(g_loss,
                                                                var_list=g_var_list)
        return d_opt, g_opt

    def sample(self, Z, reuse = True):
        g_out = self.generator(Z, reuse = reuse)
        return g_out

