import Dao_atk
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

# Xavier initialization
def xavier_init(size):
    input_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(input_dim / 2.)
    return tf.truncated_normal(shape=size, stddev=xavier_stddev)

def sample_from_noise(shape):
    return np.random.normal(0,1,size=shape)

# nosie z
G_num = tf.placeholder(dtype=tf.int32)
Z_noise = tf.placeholder(dtype=tf.float32,shape=[None,1])
#Z_label = tf.placeholder(dtype=tf.float32,shape=[None,1])
Z = Z_noise #tf.concat([Z_noise,Z_label],axis=1)

# link generator layer1
w1_link_g = tf.Variable(xavier_init([1,link_size_g['layer1']]),name='w1_link_g')
b1_link_g = tf.Variable(xavier_init([1,link_size_g['layer1']]),name='b1_link_g')
# link generator layer2
w2_link_g = tf.Variable(xavier_init([link_size_g['layer1'],link_size_g['layer2']]),name='w2_link_g')
b2_link_g = tf.Variable(xavier_init([1,link_size_g['layer2']]),name='b2_link_g')
# link generator layer3
w3_link_g = tf.Variable(xavier_init([link_size_g['layer2'],link_size_g['layer3']]),name='w3_link_g')
b3_link_g = tf.Variable(xavier_init([1,link_size_g['layer3']]),name='b3_link_g')
# link generator output
z1_link_g = tf.nn.leaky_relu(tf.add(tf.matmul(Z,w1_link_g),b1_link_g))
z2_link_g = tf.nn.leaky_relu(tf.add(tf.matmul(z1_link_g,w2_link_g),b2_link_g))
z3_link_g = tf.nn.leaky_relu(tf.add(tf.matmul(z2_link_g,w3_link_g),b3_link_g))
# link
L_g = tf.nn.l2_normalize(tf.matmul(z3_link_g,tf.transpose(z3_link_g)),axis=1) / tf.cast(G_num,tf.float32)

# embedding generator layer1
w1_embed_g = tf.Variable(xavier_init([1,embed_size_g['layer1']]),name='w1_embed_g')
b1_embed_g = tf.Variable(xavier_init([1,embed_size_g['layer1']]),name='b1_embed_g')
# embeding generator layer2
w2_embed_g = tf.Variable(xavier_init([embed_size_g['layer1'],embed_size_g['layer2']]),name='w2_embed_g')
b2_embed_g = tf.Variable(xavier_init([1,embed_size_g['layer2']]),name='b2_embed_g')
# embeding generator layer3
w3_embed_g = tf.Variable(xavier_init([embed_size_g['layer2'],embed_size_g['layer3']]),name='w3_embed_g')
b3_embed_g = tf.Variable(xavier_init([1,embed_size_g['layer3']]),name='b3_embed_g')
# embeding generator output
z1_embed_g = tf.nn.leaky_relu(tf.add(tf.matmul(Z,w1_embed_g),b1_embed_g))
z2_embed_g = tf.nn.leaky_relu(tf.add(tf.matmul(z1_embed_g,w2_embed_g),b2_embed_g))
z3_embed_g = tf.nn.leaky_relu(tf.add(tf.matmul(z2_embed_g,w3_embed_g),b3_embed_g))
# orginal embedding
E_g = z3_embed_g #tf.nn.l2_normalize(z2_embed_g,axis=1)

# GCN conv_size = 2
E_conv1_g = tf.matmul(L_g,E_g)
E_conv2_g = tf.matmul(L_g,E_conv1_g)

# embedding to rating layer1
w1_rating_g = tf.Variable(xavier_init([embed_size_g['layer3'],rating_size_g['layer1']]),name='w1_rating_g')
b1_rating_g = tf.Variable(xavier_init([1,rating_size_g['layer1']]),name='b1_rating_g')
# embedding to rating output
z1_rating_g = tf.nn.leaky_relu(tf.add(tf.matmul(E_conv2_g,w1_rating_g),b1_rating_g))
# embedding to rating
R_fake = tf.reshape(tf.reduce_mean(z1_rating_g,axis=1),[-1,1])
# input of discriminator
R_real = tf.placeholder(dtype=tf.float32,shape=[None,1])

# rating discriminator layer1
w1_rating_d = tf.Variable(xavier_init([1,rating_size_d['layer1']]),name='w1_rating_d')
b1_rating_d = tf.Variable(xavier_init([1,rating_size_d['layer1']]),name='b1_rating_d')
# rating discriminator layer2
w2_rating_d = tf.Variable(xavier_init([rating_size_d['layer1'],rating_size_d['layer2']]),name='w2_rating_d')
b2_rating_d = tf.Variable(xavier_init([1,rating_size_d['layer2']]),name='b2_rating_d')
# rating discriminator layer3
w3_rating_d = tf.Variable(xavier_init([rating_size_d['layer2'],rating_size_d['layer3']]),name='w3_rating_d')
b3_rating_d = tf.Variable(xavier_init([1,rating_size_d['layer3']]),name='b3_rating_d')
# rating discriminator layer4
w4_rating_d = tf.Variable(xavier_init([rating_size_d['layer3'],rating_size_d['layer4']]),name='w4_rating_d')
b4_rating_d = tf.Variable(xavier_init([1,rating_size_d['layer4']]),name='b4_rating_d')

def rating_discriminator(x):
    z1_rating_d = tf.nn.sigmoid(tf.add(tf.matmul(x,w1_rating_d),b1_rating_d))
    z2_rating_d = tf.nn.sigmoid(tf.add(tf.matmul(z1_rating_d,w2_rating_d),b2_rating_d))
    z3_rating_d = tf.nn.sigmoid(tf.add(tf.matmul(z2_rating_d,w3_rating_d),b3_rating_d))
    z4_rating_d = tf.nn.leaky_relu(tf.add(tf.matmul(z3_rating_d,w4_rating_d),b4_rating_d))
    return tf.reduce_mean(z4_rating_d)

# Gradient Penalty
Eps = tf.placeholder(dtype=tf.float32)
R_medium = Eps*R_real + (1-Eps)*R_fake
R_medium_d = rating_discriminator(R_medium)
# outputs of discriminator
R_real_d = rating_discriminator(R_real)
R_fake_d = rating_discriminator(R_fake)

# generator loss with Rating Penalty
loss_rating_g = -R_fake_d + reg_rating * tf.reduce_mean((R_real-R_fake)**2)
# discriminator loss
loss_rating_real_d = -R_real_d
loss_rating_fake_d = R_fake_d
loss_rating_d = loss_rating_real_d + loss_rating_fake_d + reg_grad * (tf.norm(tf.gradients(R_medium_d,R_medium),ord=2)-1)**2

# variable list
var_list_rating_g = [w1_link_g,b1_link_g,w2_link_g,b2_link_g, w1_embed_g,b1_embed_g,w2_embed_g,b2_embed_g,w1_rating_g,b1_rating_g]
var_list_rating_d = [w1_rating_d,b1_rating_d,w2_rating_d,b2_rating_d,w3_rating_d,b3_rating_d]

# optimizer for generator
rating_op_g = tf.train.AdamOptimizer(lr).minimize(loss_rating_g, var_list=var_list_rating_g)
rating_op_d = tf.train.AdamOptimizer(lr).minimize(loss_rating_d, var_list=var_list_rating_d)

saver = tf.train.Saver(max_to_keep=3)

# training
# if using the trick of cGAN, please uncomment the 'z_label_xx' codes 
# and change the sizes and weight shapes of input layers from '1 x n' into '2 x n'
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epoch_num):
        if epoch < epoch_num*0.5:
            sup_generate_num_ = int(max(sup_generate_num*0.5,inf_user_rating_num))
        elif epoch < epoch_num*0.7:
            sup_generate_num_ = int(max(sup_generate_num*0.7,inf_user_rating_num))
        else:
            sup_generate_num_ = int(sup_generate_num)

        s = time.time()
        for step in range(train_num_d):
            #print('discriminator training ' + str(step+1) + ' / ' + str(train_num_d))
            generate_num_d,batch_d,_ = dao.generateBatch(sup_generate_num_, inf_user_rating_num, selected_size)
            z_noise_d = sample_from_noise([generate_num_d,1])
            #z_label_d = np.ones([generate_num_d,1])*generate_num_d / dao.rating_len_sup
            eps_d = np.random.random()

            sess.run(rating_op_d, feed_dict={ Z_noise:z_noise_d, G_num:generate_num_d, R_real:batch_d['rating_vec'], Eps:eps_d })
    
        for step in range(train_num_g):
            #print('generator training ' + str(step+1) + ' / ' + str(train_num_g)) for bn in range(batch_num):
            generate_num_g,batch_g,_ = dao.generateBatch(sup_generate_num_, inf_user_rating_num, selected_size)
            z_noise_g = sample_from_noise([generate_num_g,1])
            #z_label_g = np.ones([generate_num_g,1])*generate_num_g / dao.rating_len_sup

            sess.run(rating_op_g, feed_dict={ Z_noise:z_noise_g, G_num:generate_num_g, R_real:batch_g['rating_vec'] })
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
            
            generate_num,batch,_ = dao.generateBatch(sup_generate_num_, inf_user_rating_num, selected_size)
            z_noise = sample_from_noise([generate_num,1])
            #z_label = np.ones([generate_num,1])*generate_num / dao.rating_len_sup
            eps = np.random.random()

            loss_rating_d_ = sess.run(loss_rating_d, feed_dict={ Z_noise:z_noise, G_num:generate_num, R_real:batch['rating_vec'], Eps:eps })
            loss_rating_g_ = sess.run(loss_rating_g, feed_dict={ Z_noise:z_noise, G_num:generate_num, R_real:batch['rating_vec'] })
        
            print('loss_rating_d ',loss_rating_d_)
            print('loss_rating_g ',loss_rating_g_)

            print(sess.run(R_fake, feed_dict={ Z_noise:z_noise, G_num:generate_num })[:20])
            #print(batch['rating_vec'])
            #print(generate_num)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
    save_path = './models_GOAT_'+dataset_name+'_'+timestamp+'/'+dataset_name+'.ckpt'
    print('saving model...')
    saver.save(sess,save_path)
    print(time_consuming)