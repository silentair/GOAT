import os
import numpy as np
import random

# Dao for GOAT
# dataset_name: Which dataset in use
# recaculate_data: Should the dao re-caculate an item-item graph and user-item interactions from current dataset
# item_rating_thre: A selected item should be rated by at least such a number of users
# sampling_size: What proportion of uesr-item-rating data should be used
class AtkDao(object):
    def __init__(self, dataset_name, item_rating_thre=8, recaculate_data=False, sampling_size=1):
        if sampling_size > 1 or sampling_size <= 0:
            print('sampling_size must in the range of (0,1]')
            raise ValueError

        '''dataset'''
        self.dataset_name = dataset_name

        '''row data'''
        self.rating_data = []       # triple tuple
        self.rating_len = None      # number of u-i pairs
        
        '''user/item/rating information'''
        self.users = []
        self.items = []
        self.user_num = 0           # user number
        self.item_num = 0           # item number
        self.user2id = {}
        self.item2id = {}
        self.id2user = {}           # No.x user has name 'xx'
        self.id2item = {}           # No.x item has name 'xx'
        self.users_RatedItem = {}   # users who rated item i and rating r
        self.items_RatedByUser = {} # items which rated by user u and rating r

        '''item-item graph'''
        self.item_graph = {}                      # key = item, val = item
        self.item_rating_thre = item_rating_thre  # a selected item at least be rated by such a number of users
        
        '''mean information'''
        self.user_means = {}        # mean values of users' ratings
        self.item_means = {}        # mean values of items' ratings
        self.global_mean = 0

        '''sample rating data'''
        self.sampling_size = sampling_size    # sampling a portion of dataset

        self.__getRatingInfo__(recaculate_data)

    def __getRatingInfo__(self, recaculate_data):
        # data path
        ratings_path = './'+ self.dataset_name + '/ratings.txt'

        # load data
        if not os.path.exists(ratings_path):
            print('dataset missing!')
            raise IOError
        
        # getting row data
        print('getting rowdata from ' + ratings_path + '...')
        with open(ratings_path) as f:
            rating_rowdata = f.read().split()
        
        # getting users&items&ratings info
        tup_len = 3                                   # data in ratings.txt is built by triple tuple
        ratings_len = len(rating_rowdata) // tup_len
        print('getting users&items&ratings from rowdata...')
        for i in range(ratings_len):
            # get user item rating
            idx = i*tup_len
            user = rating_rowdata[idx]
            item = rating_rowdata[idx + 1]
            rating = rating_rowdata[idx + 2]
            self.rating_data.append([user,item,rating])

        # sampling data
        if self.sampling_size != 1:
            print('sampling data...')
            sample_len = int(ratings_len * self.sampling_size)
            self.rating_data = random.sample(self.rating_data,sample_len)

        # user-item interactions
        interactions_path1 = './'+ self.dataset_name + '/items_RatedByUser.txt'
        interactions_path2 = './'+ self.dataset_name + '/users_RatedItem.txt'
        if recaculate_data == False and os.path.exists(interactions_path1) and os.path.exists(interactions_path2):
            print('loading user-item interactions...')
            with open(interactions_path1) as f:
                self.items_RatedByUser = eval(f.read())
            with open(interactions_path2) as f:
                self.users_RatedItem = eval(f.read())
        else:
            print('calculating user-item interactions...')
            for uir in self.rating_data:
                user = uir[0]
                item = uir[1]
                rating = uir[2]

                itemsDict = self.items_RatedByUser.get(user,{})
                itemsDict[item] = float(rating)
                self.items_RatedByUser[user] = itemsDict

                usersDict = self.users_RatedItem.get(item,{})
                usersDict[user] = float(rating)
                self.users_RatedItem[item] = usersDict
            with open(interactions_path1,'w') as f:
                f.writelines(str(self.items_RatedByUser))
            with open(interactions_path2,'w') as f:
                f.writelines(str(self.users_RatedItem))

        self.raintg_num = [len(self.items_RatedByUser[u]) for u in self.items_RatedByUser.keys()]

        # user-item
        print('getting user-item info...')
        self.users = list(self.items_RatedByUser.keys())
        self.items = list(self.users_RatedItem.keys())
        self.rating_len = len(self.rating_data)
        self.user_num = len(self.users)
        self.item_num = len(self.items)
        
        print(self.user_num,self.item_num)

        # id-name
        print('calculating id-name...')
        for n,u in enumerate(self.users):
            self.user2id[u] = n
            self.id2user[n] = u
        for n,i in enumerate(self.items):
            self.item2id[i] = n
            self.id2item[n] = i
        
        # selected items and filler items
        item_graph_path = './'+ self.dataset_name + '/item_graph.txt'
        if recaculate_data == False and os.path.exists(item_graph_path):
            print('loading selected candidate and filler candidate...')
            with open(item_graph_path) as f:
                self.item_graph = eval(f.read())
        else:
            print('calculating selected candidates and filler candidates...')
            for item in self.items:
                # selected items must be rated at least item_rating_thre times
                if len(self.users_RatedItem[item]) >= self.item_rating_thre:
                    self.item_graph[item] = []
                    for user in self.users_RatedItem[item].keys():
                        for item_u in self.items_RatedByUser[user].keys():
                            # filler items must be rated at least item_rating_thre/3 times
                            if item_u not in self.item_graph[item] and len(self.users_RatedItem[item_u]) >= self.item_rating_thre//3 and item_u != item:
                                self.item_graph[item].append(item_u)
            with open(item_graph_path,'w') as f:
                f.writelines(str(self.item_graph))

        # users' rating number
        len_uri = [len(self.items_RatedByUser[u]) for u in self.users]
        len_uri_Dict = {}
        for l in len_uri:
            if l not in len_uri_Dict.keys():
                len_uri_Dict[l] = 1
            else:
                len_uri_Dict[l] = len_uri_Dict[l] + 1
        self.rating_len_of_users_sorted_by_n = sorted(len_uri_Dict.items(),key = lambda x:x[1])
        rating_len_of_users_sorted_by_l = sorted(len_uri_Dict.items(),key = lambda x:x[0])
        accumulation = 0
        for l,n in rating_len_of_users_sorted_by_l:
            accumulation = accumulation + n
            if accumulation > self.user_num*0.5:
                self.rating_len_sup = l
                break

        # mean info
        print('getting mean from rowdata...')
        total = 0
        length = 0
        for user in self.users:
            total_u2i = sum([v for v in self.items_RatedByUser[user].values()])
            len_u2i = len(self.items_RatedByUser[user])

            if len_u2i == 0:
                self.user_means[user] = 0
            else:
                self.user_means[user] = round(2 * total_u2i / len_u2i) / 2

        for item in self.items:
            total_i2u = sum([v for v in self.users_RatedItem[item].values()])
            len_i2u = len(self.users_RatedItem[item])

            if len_u2i == 0:
                self.item_means[item] = 0
            else:
                self.item_means[item] = round(2 * total_i2u / len_i2u) / 2

            total = total + total_i2u
            length = length + len_i2u

        self.global_mean = round(2 * total / length) / 2

    # sample a sub-graph (or batch)
    # --params--
    # sup_generate_num: Max length of a fake user profile
    # inf_user_rating_num: Min length of a template user profile 
    # selected_size: Proportion of selected items
    # only_gn: Only return the length of fake user profile
    # --returns--
    # generate_num: Length of a fake user profile
    # batch: A rating vector
    # batch_items：Items corresponding to ratings
    def generateBatch(self, sup_generate_num, inf_user_rating_num, selected_size = 0.3, only_gn = False):
        # find a template user who rated at least inf_user_rating_num items
        user_s = random.sample(self.users,1)[0]
        while( len(self.items_RatedByUser[user_s]) < inf_user_rating_num):
            user_s = random.sample(self.users,1)[0]
        generate_num = min(len(self.items_RatedByUser[user_s]),sup_generate_num)

        if only_gn:
            return generate_num

        selected_num = int(generate_num * selected_size)
        filler_num = generate_num - selected_num
        # get selected items and filler items from template
        selected_candidates = []
        filler_candidates = []
        for item_u in self.items_RatedByUser[user_s].keys():
            if item_u in self.item_graph.keys():
                selected_candidates.append(item_u)
            elif len(self.users_RatedItem[item_u]) > self.item_rating_thre//3:
                filler_candidates.append(item_u)

        # fill-up/cut-off selected candidates if insufficient/overflow
        if len(selected_candidates) < selected_num:
            other_selected_candidates = self.item_graph.keys()-selected_candidates
            other_selected_num = selected_num - len(selected_candidates)
            selected_items = selected_candidates + random.sample(other_selected_candidates,other_selected_num)
        else:
            selected_items = random.sample(selected_candidates,selected_num)

        # fill-up/cut-off filler candidates if insufficient/overflow
        if len(filler_candidates) < filler_num:
            other_filler_candidates = []
            other_filler_num = filler_num - len(filler_candidates)
            # filler items are sampled from those co-rated with selected items
            for si in selected_items:
                for candidate in self.item_graph[si]:
                    if candidate not in selected_items and candidate not in filler_candidates and candidate not in other_filler_candidates:
                        other_filler_candidates.append(candidate)
            # no enough filler candidates, usually won't happen
            if other_filler_candidates == []:
                other_filler_candidates = set(self.items) - self.item_graph.keys() - set(filler_candidates)
                print('Ooops',other_filler_num)
            filler_items = filler_candidates + random.sample(other_filler_candidates,other_filler_num)
        else:
            filler_items = random.sample(filler_candidates,filler_num)
       
        # generate one batch
        batch = {}
        batch_items = selected_items + filler_items
        batch_ids = [self.item2id[i] for i in batch_items]

        batch['rating_vec'] = np.array([[self.item_means[self.id2item[id]]] for id in batch_ids])

        return generate_num,batch,batch_items

    # sample an entry rating vector (or batch)
    # --params--
    # None
    # --returns--
    # batch: A rating vector filled with a lot of zero ratings
    def generateBatch_naive(self):
        user_s = random.sample(self.users,1)[0]
        while( len(self.items_RatedByUser[user_s]) < self.rating_len_of_users_sorted_by_n[-10][0]):
            user_s = random.sample(self.users,1)[0]
       
        # generate batch
        batch = {}
        batch_items = self.items_RatedByUser[user_s].keys()
        batch_ids = [self.item2id[i] for i in batch_items]

        # rescale ratings from [0, 5] into [-1, 1]
        rating_vec = np.ones([1,len(self.items)])*-1
        for iid in batch_ids:
            rating_vec[0][iid] = self.items_RatedByUser[user_s][self.id2item[iid]]*0.4 - 1

        batch['rating_vec'] = rating_vec

        return batch