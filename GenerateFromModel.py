import Dao_atk
import numpy as np
import random
import tensorflow.compat.v1 as tf
from tqdm import tqdm
from functools import partial

def sample_from_noise(shape):
    return np.random.normal(0,1,size=shape)

def map_rating(rating, mx=5.0, mn=1.0):
    rating = max(min(rating,mx),mn)
    #return round(rating)
    return round(rating*2)/2
    #return round(rating*5)/5
    #return round(rating*10)/10

##########################
# tensorflow 1.x version #
##########################
def generateFromModel_tf1(dataset_name, model_path, inject_size, sup_generate_num, inf_user_rating_num, selected_size):
    # preparing
    dao = Dao_atk.AtkDao(dataset_name)
    inject_num = int(dao.user_num * inject_size)

    # load model
    saver = tf.train.import_meta_graph(model_path+dataset_name+'.ckpt.meta')
    with tf.Session() as sess:
        saver.restore(sess,model_path+dataset_name+'.ckpt')
            
        G_num = tf.placeholder(dtype=tf.int32)
        Z_noise = tf.placeholder(dtype=tf.float32,shape=[None,1])
        #Z_label = tf.placeholder(dtype=tf.float32,shape=[None,1])
        Z = Z_noise#tf.concat([Z_noise,Z_label],axis=1)
        
        # link generator layer1
        w1_link_g = tf.get_default_graph().get_tensor_by_name('w1_link_g:0')
        b1_link_g = tf.get_default_graph().get_tensor_by_name('b1_link_g:0')
        # link generator layer2
        w2_link_g = tf.get_default_graph().get_tensor_by_name('w2_link_g:0')
        b2_link_g = tf.get_default_graph().get_tensor_by_name('b2_link_g:0')
        # link generator layer3
        w3_link_g = tf.get_default_graph().get_tensor_by_name('w3_link_g:0')
        b3_link_g = tf.get_default_graph().get_tensor_by_name('b3_link_g:0')
        # link generator output
        z1_link_g = tf.nn.leaky_relu(tf.add(tf.matmul(Z,w1_link_g),b1_link_g))
        z2_link_g = tf.nn.leaky_relu(tf.add(tf.matmul(z1_link_g,w2_link_g),b2_link_g))
        z3_link_g = tf.nn.leaky_relu(tf.add(tf.matmul(z2_link_g,w3_link_g),b3_link_g))
        # link
        L_g = tf.nn.l2_normalize(tf.matmul(z3_link_g,tf.transpose(z3_link_g)),axis=1) / tf.cast(G_num,tf.float32)
        
        # embedding generator layer1
        w1_embed_g = tf.get_default_graph().get_tensor_by_name('w1_embed_g:0')
        b1_embed_g = tf.get_default_graph().get_tensor_by_name('b1_embed_g:0')
        # embeding generator layer2
        w2_embed_g = tf.get_default_graph().get_tensor_by_name('w2_embed_g:0')
        b2_embed_g = tf.get_default_graph().get_tensor_by_name('b2_embed_g:0')
        # embeding generator layer3
        w3_embed_g = tf.get_default_graph().get_tensor_by_name('w3_embed_g:0')
        b3_embed_g = tf.get_default_graph().get_tensor_by_name('b3_embed_g:0')
        # embeding generator output
        z1_embed_g = tf.nn.leaky_relu(tf.add(tf.matmul(Z,w1_embed_g),b1_embed_g))
        z2_embed_g = tf.nn.leaky_relu(tf.add(tf.matmul(z1_embed_g,w2_embed_g),b2_embed_g))
        z3_embed_g = tf.nn.leaky_relu(tf.add(tf.matmul(z2_embed_g,w3_embed_g),b3_embed_g))
        # orginal embedding
        E_g = z3_embed_g
        
        # GCN conv_size = 2
        E_conv1_g = tf.matmul(L_g,E_g)
        E_conv2_g = tf.matmul(L_g,E_conv1_g)
        
        # embedding to rating layer1
        w1_rating_g = tf.get_default_graph().get_tensor_by_name('w1_rating_g:0')
        b1_rating_g = tf.get_default_graph().get_tensor_by_name('b1_rating_g:0')
        # embedding to rating output
        z1_rating_g = tf.nn.leaky_relu(tf.add(tf.matmul(E_conv2_g,w1_rating_g),b1_rating_g))
        # embedding to rating
        R_fake = tf.reshape(tf.reduce_mean(z1_rating_g,axis=1),[-1,1])

        # load target items
        with open('./'+dataset_name+'/push_target_items.txt') as f:
            targets = f.read().split()
        target_rating = 4.0

        fake_uir = ''
        # generate fake user profile with target items
        print('generating fake users...')
        for n in tqdm(range(inject_num)):
            generate_num,_,items = dao.generateBatch(sup_generate_num, inf_user_rating_num, selected_size)
            z_noise = sample_from_noise([generate_num,1])
            #z_label = np.ones([generate_num,1])*generate_num / dao.rating_len_sup
            rating_vec = sess.run(R_fake, feed_dict={Z_noise:z_noise, G_num:generate_num}).flatten().tolist()
            # discard abnormal profiles (optional)
            #while(np.average(rating_vec) > 4.5 or np.average(rating_vec) < 3):
            #    generate_num,_,items = dao.generateBatch(sup_generate_num, inf_user_rating_num)
            #    z_noise = sample_from_noise([generate_num,1])
            #    #z_label = np.ones([generate_num,1])*generate_num / dao.rating_len_sup
            #    rating_vec = sess.run(R_fake, feed_dict={Z_noise:z_noise, G_num:generate_num}).flatten().tolist()

            u = 'fake_'+str(n)
            for i,r in zip(items,rating_vec):
                r = map_rating(r)
                fake_uir += u+' '+i+' '+str(r)+'\n'
            for t in targets:
                fake_uir += u+' '+t+' '+str(target_rating)+'\n'
            
    # save to .txt
    print('saving fake ratings...')
    with open('./'+dataset_name+'/fake_ratings.txt','w') as f:
        f.writelines(fake_uir)

# using example for above function
#dataset_name = 'xxx'
#model_path = 'models_GOAT_xxx'
#inject_size = 0.01
#sup_generate_num = 15
#inf_user_rating_num = 10
#selected_size = 0.3
#generateFromModel_tf1(dataset_name, model_path, inject_size, sup_generate_num, inf_user_rating_num, selected_size)



##########################
# tensorflow 2.x version #
##########################
# generator network
link_size_g = {'layer1': 128, 'layer2': 256, 'layer3':32}
embed_size_g = {'layer1':128, 'layer2':256, 'layer3':64}
rating_size_g = {'layer1': 128 }
# discriminator network
rating_size_d = {'layer1':1024, 'layer2':512, 'layer3':256, 'layer4':1}

# model
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
        self.optimizer_g = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.optimizer_d = tf.keras.optimizers.Adam(learning_rate=0.001)

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

def generateFromModel_tf2(dataset_name, model_path, inject_size, sup_generate_num, inf_user_rating_num, selected_size):
    # preparing
    dao = Dao_atk.AtkDao(dataset_name)
    inject_num = int(dao.user_num * inject_size)

    attacker = AttackGenerator()
    attacker.load_weights(model_path).expect_partial()

    # load target items
    with open('./'+dataset_name+'/push_target_items.txt') as f:
        targets = f.read().split()
    target_rating = 4.0

    fake_uir = ''
    # generate fake user profile with target items
    print('generating fake users...')
    for n in tqdm(range(inject_num)):
        generate_num,_,items = dao.generateBatch(sup_generate_num, inf_user_rating_num, selected_size)
        z_noise = sample_from_noise([generate_num,1])
        #z_label = np.ones([generate_num,1])*generate_num / dao.rating_len_sup
        rating_vec = attacker(z_noise,generate_num).numpy().flatten().tolist()
        # discard abnormal profiles (optional)
        #while(np.average(rating_vec) > 4.5 or np.average(rating_vec) < 3):
        #    generate_num,_,items = dao.generateBatch(sup_generate_num, inf_user_rating_num)
        #    z_noise = sample_from_noise([generate_num,1])
        #    #z_label = np.ones([generate_num,1])*generate_num / dao.rating_len_sup
        #rating_vec = attacker(z_noise,generate_num).numpy().flatten().tolist()

        u = 'fake_'+str(n)
        for i,r in zip(items,rating_vec):
            r = map_rating(r)
            fake_uir += u+' '+i+' '+str(r)+'\n'
        for t in targets:
            fake_uir += u+' '+t+' '+str(target_rating)+'\n'
            
    # save to .txt
    print('saving fake ratings...')
    with open('./'+dataset_name+'/fake_ratings.txt','w') as f:
        f.writelines(fake_uir)

# using example for above function
#dataset_name = 'xxx'
#model_path = 'models_GOAT_xxx/xxx.weights'
#inject_size = 0.01
#sup_generate_num = 15
#inf_user_rating_num = 10
#selected_size = 0.3
#generateFromModel_tf2(dataset_name, model_path, inject_size, sup_generate_num, inf_user_rating_num, selected_size)