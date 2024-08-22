from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import random
from itertools import chain, repeat
class mydataset(Dataset):
    def __init__(self,train_matrix, num_poi,num_user, negative_num):
        self.train_matrix = train_matrix
        self.num_poi = num_poi
        self.num_user = num_user
        self.negative_num = negative_num
        self.all_data_len = int(train_matrix.sum())
        # self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.indices = np.zeros([self.all_data_len],dtype=np.int64)
        self.user_indices = np.zeros([self.all_data_len],dtype=np.int64)
        idx = 0
        for uid in range(num_user):
            row = self.train_matrix.getrow(uid).indices.tolist()
            for i in range(len(row)):
                self.indices[idx+i] = i
                self.user_indices[idx+i] = uid
            idx += len(row)
    def __len__(self):
        return self.all_data_len

    def __getitem__(self, idx):
        item_list = np.arange(self.num_poi).tolist()
        row = self.train_matrix.getrow(self.user_indices[idx]).indices.tolist()
        positive = [row[self.indices[idx]]]
        # random.shuffle(positives)
        histories = np.array(row)

        negative = list(set(item_list)-set(row))
        random.shuffle(negative)

        negative = negative[:len(positive)*self.negative_num]
        negatives = np.array(negative).reshape([-1,self.negative_num])

        a = np.array(positive).reshape(-1,1)
        data = np.concatenate((a, negatives),axis=-1)
        data = data.reshape(-1)

        positive_label = np.array([1]).repeat(len(positive)).reshape(-1,1)
        negative_label = np.array([0]).repeat(len(positive)*self.negative_num).reshape(-1,self.negative_num)
        labels = np.concatenate((positive_label,negative_label),axis=-1).reshape(-1)
        
        user_history = histories
        train_data = data
        train_label = torch.tensor(labels,dtype=torch.float32)

        user_history=torch.LongTensor(user_history)
        train_data=torch.LongTensor(train_data)

        return user_history, train_data, train_label
        
    def my_collate_fn(self,samples):
        collate_history = []
        collate_target = []
        collate_label = []
        max_len = 0
        for sample in samples:
            max_len = max(len(sample[0]),max_len)
        for sample in samples:
            padding_tensor = torch.zeros(max_len-len(sample[0]),dtype=torch.int64)+(self.num_poi)
            t = torch.cat([sample[0],padding_tensor],axis = 0)
            # collate_history.append(repeat(t),self.negative_num+1)
            for i in range(self.negative_num+1):
                collate_history.append(t)
            collate_target.append(sample[1])
            collate_label.append(sample[2])
        return torch.stack(collate_history), torch.cat(collate_target), torch.cat(collate_label)