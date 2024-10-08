import scipy.sparse as sparse
import numpy as np
import random
from haversine import haversine_vector

def train_test_val_split_with_time(place_list, freq_list, time_list, test_size, val_size):
    li = []
    for i in range(len(place_list)):
        li.append((place_list[i], time_list[i], freq_list[i]))
    li.sort(key=lambda x:-x[1])
    test = li[:int(len(li)*test_size)]
    train_ = li[int(len(li)*test_size):]

    val_num = int(len(li)*val_size)
    if val_num == 0:
        val_num=1
    val = train_[:val_num]
    train = train_[val_num:]

    random.shuffle(train)
    test_place = []
    test_freq = []
    for i in test:
        test_place.append(i[0])
        test_freq.append(i[2])

    train_place=[]
    train_freq=[]
    for i in train:
        train_place.append(i[0])
        train_freq.append(i[2])

    val_place = []
    val_freq = []

    for i in val:
        val_place.append(i[0])
        val_freq.append(i[2])
    return train_place, test_place, val_place, train_freq, test_freq, val_freq

class Tokyo(object):
    def __init__(self):
        self.user_num = 3725
        self.poi_num = 10768
        self.directory_path = "./data/Tokyo/"
        self.checkin_file = 'checkins.txt'
        self.poi_file = 'poi_coos.txt'
    def read_raw_data(self):
        all_data = open(self.directory_path + self.checkin_file, 'r').readlines()
        sparse_raw_matrix = sparse.dok_matrix((self.user_num, self.poi_num))
        sparse_raw_time_matrix= sparse.dok_matrix((self.user_num, self.poi_num))
        for eachline in all_data:
            uid, lid, time = eachline.strip().split()
            uid, lid, time = int(uid), int(lid), float(time)
            sparse_raw_matrix[uid, lid] = sparse_raw_matrix[uid, lid] + 1
            if sparse_raw_time_matrix[uid,lid] < time:
                sparse_raw_time_matrix[uid, lid] = time
        return sparse_raw_matrix.tocsr(), sparse_raw_time_matrix.tocsr()

    def split_data(self, raw_matrix, time_matrix, random_seed=0):
        test_size = 0.2
        val_size = 0.1
        train_matrix = sparse.dok_matrix((self.user_num, self.poi_num))
        
        val_positive = []
        test_positive = []
        self.POI_POI_Graph = np.zeros([self.poi_num,self.poi_num])
        self.user_POI_Graph = np.zeros([self.user_num,self.poi_num])
        
        pois = set(range(self.poi_num))
        for user_id in range(self.user_num):
            place_list = raw_matrix.getrow(user_id).indices
            freq_list = raw_matrix.getrow(user_id).data
            time_list = time_matrix.getrow(user_id).data

            train_place, test_place, val_place, train_freq, test_freq, val_freq = train_test_val_split_with_time(place_list, freq_list, time_list, test_size, val_size)
            
            for i in range(len(train_place)):
                train_matrix[user_id, train_place[i]] = train_freq[i]

                self.user_POI_Graph[user_id][train_place[i]] = 1

                if i <len(train_place)-1:
                    self.POI_POI_Graph[train_place[i]][train_place[i+1]] +=1
            test_positive.append(test_place)
            val_positive.append(val_place)

        return train_matrix.tocsr(), test_positive, val_positive

    def read_poi_coos(self,near_POI_num):
        poi_coos = {}
        poi_data = open(self.directory_path + self.poi_file, 'r').readlines()
        for eachline in poi_data:
            lid, lat, lng = eachline.strip().split()
            lid, lat, lng = int(lid), float(lat), float(lng)
            poi_coos[lid] = (lat, lng)

        place_coords = []
        for k, v in poi_coos.items():
            place_coords.append([v[0], v[1]])
        self.place_coos = place_coords
        self.dist_matrix = np.array(haversine_vector(place_coords,place_coords,comb=True))
        self.nearPOI = np.argpartition(self.dist_matrix,near_POI_num)[:,:near_POI_num]
        
        return place_coords

    def generate_data(self, random_seed, near_POI_num):
        raw_matrix, time_matrix = self.read_raw_data()
        train_matrix, test_positive, val_positive = self.split_data(raw_matrix, time_matrix, random_seed)
        place_coords =self.read_poi_coos(near_POI_num)
        return train_matrix, test_positive, val_positive, place_coords

class Philadelphia(object):
    def __init__(self):
        self.user_num = 15359
        self.poi_num = 14586
        self.directory_path = "./data/Philadelphia/"
        self.checkin_file = 'checkins.txt'
        self.poi_file = 'poi_coos.txt'
    def read_raw_data(self):
        all_data = open(self.directory_path + self.checkin_file, 'r').readlines()
        sparse_raw_matrix = sparse.dok_matrix((self.user_num, self.poi_num))
        sparse_raw_time_matrix= sparse.dok_matrix((self.user_num, self.poi_num))
        for eachline in all_data:
            uid, lid, time = eachline.strip().split()
            uid, lid, time = int(uid), int(lid), float(time)
            sparse_raw_matrix[uid, lid] = sparse_raw_matrix[uid, lid] + 1
            if sparse_raw_time_matrix[uid,lid] < time:
                sparse_raw_time_matrix[uid, lid] = time
        return sparse_raw_matrix.tocsr(), sparse_raw_time_matrix.tocsr()

    def split_data(self, raw_matrix, time_matrix, random_seed=0):
        test_size = 0.2
        val_size = 0.1
        train_matrix = sparse.dok_matrix((self.user_num, self.poi_num))
        
        val_positive = []
        test_positive = []
        self.POI_POI_Graph = np.zeros([self.poi_num,self.poi_num])
        self.user_POI_Graph = np.zeros([self.user_num,self.poi_num])
        
        pois = set(range(self.poi_num))
        for user_id in range(self.user_num):
            place_list = raw_matrix.getrow(user_id).indices
            freq_list = raw_matrix.getrow(user_id).data
            time_list = time_matrix.getrow(user_id).data

            train_place, test_place, val_place, train_freq, test_freq, val_freq = train_test_val_split_with_time(place_list, freq_list, time_list, test_size, val_size)
            
            for i in range(len(train_place)):
                train_matrix[user_id, train_place[i]] = train_freq[i]

                self.user_POI_Graph[user_id][train_place[i]] = 1

                if i <len(train_place)-1:
                    self.POI_POI_Graph[train_place[i]][train_place[i+1]] +=1
            test_positive.append(test_place)
            val_positive.append(val_place)

        return train_matrix.tocsr(), test_positive, val_positive

    def read_poi_coos(self,near_POI_num):
        poi_coos = {}
        poi_data = open(self.directory_path + self.poi_file, 'r').readlines()
        for eachline in poi_data:
            lid, lat, lng = eachline.strip().split()
            lid, lat, lng = int(lid), float(lat), float(lng)
            poi_coos[lid] = (lat, lng)

        place_coords = []
        for k, v in poi_coos.items():
            place_coords.append([v[0], v[1]])
        self.place_coos = place_coords
        self.dist_matrix = np.array(haversine_vector(place_coords,place_coords,comb=True))
        self.nearPOI = np.argpartition(self.dist_matrix,near_POI_num)[:,:near_POI_num]
        
        return place_coords

    def generate_data(self, random_seed, near_POI_num):
        raw_matrix, time_matrix = self.read_raw_data()
        train_matrix, test_positive, val_positive = self.split_data(raw_matrix, time_matrix, random_seed)
        place_coords =self.read_poi_coos(near_POI_num)
        return train_matrix, test_positive, val_positive, place_coords
    
class NewYork(object):
    def __init__(self):
        self.user_num = 6638
        self.poi_num = 21102
        self.directory_path = "./data/NewYork/"
        self.checkin_file = 'checkins.txt'
        self.poi_file = 'poi_coos.txt'
    def read_raw_data(self):
        all_data = open(self.directory_path + self.checkin_file, 'r').readlines()
        sparse_raw_matrix = sparse.dok_matrix((self.user_num, self.poi_num))
        sparse_raw_time_matrix= sparse.dok_matrix((self.user_num, self.poi_num))
        for eachline in all_data:
            uid, lid, time = eachline.strip().split()
            uid, lid, time = int(uid), int(lid), float(time)
            sparse_raw_matrix[uid, lid] = sparse_raw_matrix[uid, lid] + 1
            if sparse_raw_time_matrix[uid,lid] < time:
                sparse_raw_time_matrix[uid, lid] = time
        return sparse_raw_matrix.tocsr(), sparse_raw_time_matrix.tocsr()

    def split_data(self, raw_matrix, time_matrix, random_seed=0):
        test_size = 0.2
        val_size = 0.1
        train_matrix = sparse.dok_matrix((self.user_num, self.poi_num))
        
        val_positive = []
        test_positive = []
        self.POI_POI_Graph = np.zeros([self.poi_num,self.poi_num])
        self.user_POI_Graph = np.zeros([self.user_num,self.poi_num])
        
        pois = set(range(self.poi_num))
        for user_id in range(self.user_num):
            place_list = raw_matrix.getrow(user_id).indices
            freq_list = raw_matrix.getrow(user_id).data
            time_list = time_matrix.getrow(user_id).data

            train_place, test_place, val_place, train_freq, test_freq, val_freq = train_test_val_split_with_time(place_list, freq_list, time_list, test_size, val_size)
            
            for i in range(len(train_place)):
                train_matrix[user_id, train_place[i]] = train_freq[i]

                self.user_POI_Graph[user_id][train_place[i]] = 1

                if i <len(train_place)-1:
                    self.POI_POI_Graph[train_place[i]][train_place[i+1]] +=1
            test_positive.append(test_place)
            val_positive.append(val_place)

        return train_matrix.tocsr(), test_positive, val_positive

    def read_poi_coos(self,near_POI_num):
        poi_coos = {}
        poi_data = open(self.directory_path + self.poi_file, 'r').readlines()
        for eachline in poi_data:
            lid, lat, lng = eachline.strip().split()
            lid, lat, lng = int(lid), float(lat), float(lng)
            poi_coos[lid] = (lat, lng)

        place_coords = []
        for k, v in poi_coos.items():
            place_coords.append([v[0], v[1]])
        self.place_coos = place_coords
        self.dist_matrix = np.array(haversine_vector(place_coords,place_coords,comb=True))
        self.nearPOI = np.argpartition(self.dist_matrix,near_POI_num)[:,:near_POI_num]
        
        return place_coords

    def generate_data(self, random_seed, near_POI_num):
        raw_matrix, time_matrix = self.read_raw_data()
        train_matrix, test_positive, val_positive = self.split_data(raw_matrix, time_matrix, random_seed)
        place_coords =self.read_poi_coos(near_POI_num)
        return train_matrix, test_positive, val_positive, place_coords
    