import Dao_atk
from functools import partial
import numpy as np
import tensorflow as tf
import time

dataset_name = 'DouBan_small'
dao = Dao_atk.AtkDao(dataset_name)

# generator network
link_size_g = {'layer1': 128, 'layer2': 256, 'layer3':32}
embed_size_g = {'layer1':128, 'layer2':256, 'layer3':64}
rating_size_g = {'layer1': 128 }
# discriminator network
rating_size_d = {'layer1':1024, 'layer2':512, 'layer3':256, 'layer4':1}

# max training epochs
epoch_num = 10000
# discriminator training number in each epoch
train_num_d = 5
# generator training number in each epoch
train_num_g = 10
# max length of a fake user profile
sup_generate_num = dao.rating_len_sup
# min length of a template user profile 
inf_user_rating_num = dao.rating_len_of_users_sorted_by_n[-10][0]
# size of selected items
selected_size = 0.3
# regularization
reg_grad = 10
reg_rating = 10
# learning rate
lr = 0.001

# time
time_consuming = 0

def sample_from_noise(shape):
    return np.random.normal(0,1,size=shape)

# build model
class AttackGenerator(tf.keras.Model):
    def __init__(self, name='AttackGenerator', **kwargs):
        super(AttackGenerator, self).__init__(name=name, **kwargs)
        ''' link generator '''
        # link generator layer1
        self.link_g_layer1 = tf.keras.layers.Dense(link_size_g['layer1'], activation=partial(tf.nn.leaky_relu, alpha=0.2))
        # link generator layer2
        self.link_g_layer2 = tf.keras.layers.Dense(link_size_g['layer2'], activation=partial(tf.nn.leaky_relu, alpha=0.2))
        # link generator layer3
        self.link_g_layer3 = tf.keras.layers.Dense(link_size_g['layer3'], activation=partial(tf.nn.leaky_relu, alpha=0.2))
        ''' embedding generator '''
        # embedding generator layer1
        self.embed_g_layer1 = tf.keras.layers.Dense(embed_size_g['layer1'], activation=partial(tf.nn.leaky_relu, alpha=0.2))
        # embedding generator layer2
        self.embed_g_layer2 = tf.keras.layers.Dense(embed_size_g['layer2'], activation=partial(tf.nn.leaky_relu, alpha=0.2))
        # embedding generator layer3
        self.embed_g_layer3 = tf.keras.layers.Dense(embed_size_g['layer3'], activation=partial(tf.nn.leaky_relu, alpha=0.2))
        ''' rating generator '''
        # rating generator layer1
        self.rating_g_layer1 = tf.keras.layers.Dense(rating_size_g['layer1'], activation=partial(tf.nn.leaky_relu, alpha=0.2))
        ''' rating discriminator '''
        # rating discriminator layer1
        self.rating_d_layer1 = tf.keras.layers.Dense(rating_size_d['layer1'], activation='sigmoid')
        # rating discriminator layer2
        self.rating_d_layer2 = tf.keras.layers.Dense(rating_size_d['layer2'], activation='sigmoid')
        # rating discriminator layer3
        self.rating_d_layer3 = tf.keras.layers.Dense(rating_size_d['layer3'], activation='sigmoid')
        # rating discriminator layer4
        self.rating_d_layer4 = tf.keras.layers.Dense(rating_size_d['layer4'], activation=partial(tf.nn.leaky_relu, alpha=0.2))
        ''' optimizer '''
        self.optimizer_g = tf.keras.optimizers.Adam(learning_rate=lr)
        self.optimizer_d = tf.keras.optimizers.Adam(learning_rate=lr)

    # generator
    def call(self, Z, G_num):
        ''' link generator output '''
        z1_link_g = self.link_g_layer1(Z)
        z2_link_g = self.link_g_layer2(z1_link_g)
        z3_link_g = self.link_g_layer3(z2_link_g)
        L_g = tf.nn.l2_normalize(tf.matmul(z3_link_g,tf.transpose(z3_link_g)),axis=1) / tf.cast(G_num,tf.float32)
        ''' embeding generator output '''
        z1_embed_g = self.embed_g_layer1(Z)
        z2_embed_g = self.embed_g_layer2(z1_embed_g)
        z3_embed_g = self.embed_g_layer3(z2_embed_g)
        E_g = z3_embed_g
        # GCN conv_size = 2
        E_conv1_g = tf.matmul(L_g,E_g)
        E_conv2_g = tf.matmul(L_g,E_conv1_g)
        ''' embedding to rating output '''
        z1_rating_g = self.rating_g_layer1(E_conv2_g)
        # embedding to rating
        R_fake = tf.reshape(tf.reduce_mean(z1_rating_g,axis=1),[-1,1])
        return R_fake

    # discriminator
    def discriminator_call(self, X):
        z1_rating_d = self.rating_d_layer1(X)
        z2_rating_d = self.rating_d_layer2(z1_rating_d)
        z3_rating_d = self.rating_d_layer3(z2_rating_d)
        z4_rating_d = self.rating_d_layer4(z3_rating_d)
        return tf.reduce_mean(z4_rating_d)

    # generator training
    def train_g(self, Z, G_num, R_real):
        # GradientTape
        with tf.GradientTape() as tape:
            # generator loss
            loss_rating_g = self.loss_g(Z, G_num, R_real)
            
        # variable list
        var_list_rating_g = [self.link_g_layer1.kernel,self.link_g_layer1.bias,self.link_g_layer2.kernel,self.link_g_layer2.bias,self.link_g_layer3.kernel,self.link_g_layer3.bias, \
                            self.embed_g_layer1.kernel,self.embed_g_layer1.bias,self.embed_g_layer2.kernel,self.embed_g_layer2.bias,self.embed_g_layer3.kernel,self.embed_g_layer3.bias, \
                            self.rating_g_layer1.kernel,self.rating_g_layer1.bias]
        # gradient
        grads = tape.gradient(loss_rating_g, var_list_rating_g)
        # minimize
        self.optimizer_g.apply_gradients(zip(grads, var_list_rating_g))

    # generator loss
    def loss_g(self, Z, G_num, R_real):
        # generated ratings
        R_fake = self.call(Z, G_num)
        # discrimation
        R_fake_d = self.discriminator_call(R_fake)
        # generator loss
        loss_rating_g = -R_fake_d + reg_rating * tf.reduce_mean((R_real-R_fake)**2)
        return loss_rating_g

    # discriminator training
    def train_d(self, Z, G_num, R_real, Eps):
        # GradientTape
        with tf.GradientTape() as tape:
            # discriminator loss
            loss_rating_d = self.loss_d(Z, G_num, R_real, Eps)

        # variable list
        var_list_rating_d = [self.rating_d_layer1.kernel,self.rating_d_layer1.bias,self.rating_d_layer2.kernel,self.rating_d_layer2.bias,self.rating_d_layer3.kernel,self.rating_d_layer3.bias,self.rating_d_layer4.kernel,self.rating_d_layer4.bias]
        # gradient
        grads = tape.gradient(loss_rating_d, var_list_rating_d)
        # minimize
        self.optimizer_d.apply_gradients(zip(grads, var_list_rating_d))

    # discriminator loss
    def loss_d(self, Z, G_num, R_real, Eps):
            # generated ratings
            R_fake = self.call(Z, G_num)
            # Gradient Penalty X_hat
            R_medium = Eps*R_real + (1-Eps)*R_fake
            # discrimation
            R_real_d = self.discriminator_call(R_real)
            R_fake_d = self.discriminator_call(R_fake)
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(R_medium)
                R_medium_d = self.discriminator_call(R_medium)
            # discriminator loss
            loss_rating_real_d = -R_real_d
            loss_rating_fake_d = R_fake_d
            R_medium_grad = tape.gradient(R_medium_d,R_medium)
            loss_rating_d = loss_rating_real_d + loss_rating_fake_d + reg_grad * (tf.norm(R_medium_grad,ord=2)-1)**2
            return loss_rating_d

if __name__ == '__main__':
    attacker = AttackGenerator()
    
    for epoch in range(epoch_num):
        if epoch < epoch_num*0.5:
            sup_generate_num_ = int(sup_generate_num*0.5)
        elif epoch < epoch_num*0.7:
            sup_generate_num_ = int(sup_generate_num*0.7)
        else:
            sup_generate_num_ = int(sup_generate_num)

        s = time.time()
        for step in range(train_num_d):
            #print('discriminator training ' + str(step+1) + ' / ' + str(train_num_d))
            generate_num_d,batch_d,_ = dao.generateBatch(sup_generate_num, inf_user_rating_num, selected_size)
            z_noise_d = sample_from_noise([generate_num_d,1])
            #z_label_d = np.ones([generate_num_d,1])*generate_num_d / dao.rating_len_sup
            z = z_noise_d#z = np.concatenate([z_noise_d,z_label_d],1)
            eps_d = np.random.random()

            attacker.train_d(z,generate_num_d,batch_d['rating_vec'],eps_d)
            
        for step in range(train_num_g):
            #print('generator training ' + str(step+1) + ' / ' + str(train_num_g)) for bn in range(batch_num):
            generate_num_g,batch_g,_ = dao.generateBatch(sup_generate_num, inf_user_rating_num, selected_size)
            z_noise_g = sample_from_noise([generate_num_g,1])
            #z_label_g = np.ones([generate_num_g,1])*generate_num_g / dao.rating_len_sup
            z = z_noise_g#z = np.concatenate([z_noise_g,z_label_g],1)

            attacker.train_g(z,generate_num_g,batch_g['rating_vec'])
        e = time.time()
        
        time_consuming = time_consuming + e - s
        # convergence
        # type your convergence condition here if needed
        #if xxxx:
        #    print('epoch '+str(epoch+1)+' / '+str(epoch_num))
        #    print('loss_rating_d ',loss_rating_d_)
        #    print('loss_rating_g ',loss_rating_g_)
        #    print(sess.run(R_fake, feed_dict={ Z_noise:z_noise, Z_label:z_label, G_num:generate_num }))
        #    print(batch['rating_vec'])
        #    print(generate_num)
        #    break

        if epoch % 10 == 0:
            print('epoch '+str(epoch+1)+' / '+str(epoch_num))
            
            generate_num,batch,_ = dao.generateBatch(sup_generate_num, inf_user_rating_num, selected_size)
            z_noise = sample_from_noise([generate_num,1])
            #z_label = np.ones([generate_num,1])*generate_num / dao.rating_len_sup
            z = z_noise#z = np.concatenate([z_noise,z_label],1)
            eps = np.random.random()

            loss_rating_d_ = attacker.loss_d(z,generate_num,batch['rating_vec'],eps)
            loss_rating_g_ = attacker.loss_g(z,generate_num,batch['rating_vec'])
            print('loss_rating_d ',loss_rating_d_)
            print('loss_rating_g ',loss_rating_g_)

            R_fake = attacker.call(z,generate_num)
            print('fake_rating',R_fake[:20])
            #print(batch['rating_vec'])
            #print(generate_num)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
    save_path = './models_GOAT_'+dataset_name+'_'+timestamp+'/'+dataset_name+'.weights'

    print('saving model...')
    attacker.save_weights(save_path)
    print(time_consuming)
