import random
import numpy as np
import torch

def train_batch(train_matrix, num_poi, uid, negative_num):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    item_list = np.arange(num_poi).tolist()

    positives = train_matrix.getrow(uid).indices.tolist()
    random.shuffle(positives)
    histories = np.array([positives])

    negative = list(set(item_list)-set(positives))
    random.shuffle(negative)
    nidx = np.random.randint(0,len(negative),size=len(positives)*negative_num)
    negative = np.array(negative)[nidx]
    negatives = np.array(negative).reshape([-1,negative_num])

    a= np.array(positives).reshape(-1,1)
    data = np.concatenate((a, negatives),axis=-1)
    data = data.reshape(-1)

    positive_label = np.array([1]).repeat(len(positives)).reshape(-1,1)
    negative_label = np.array([0]).repeat(len(positives)*negative_num).reshape(-1,negative_num)
    labels = np.concatenate((positive_label,negative_label),axis=-1).reshape(-1)
    
    user_history = histories
    train_data = data
    train_label = torch.tensor(labels,dtype=torch.float32).to(DEVICE)

    user_history=torch.LongTensor(user_history).to(DEVICE)
    train_data=torch.LongTensor(train_data).to(DEVICE)

    return user_history, train_data, train_label

def test_batch(train_matrix, uid):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    history = train_matrix.getrow(uid).indices.tolist()
    negative = list(set(range(train_matrix.shape[1]))-set(history))
    # histories = np.array([history]).repeat(len(negative),axis=0)
    histories = np.array([history])


    negative_label = np.array([0]).repeat(len(negative))

    user_history = histories
    train_data = negative
    train_label = torch.tensor(negative_label,dtype=torch.float32).to(DEVICE)

    user_history=torch.LongTensor(user_history).to(DEVICE)
    train_data=torch.LongTensor(train_data).to(DEVICE)
    
    return user_history, train_data, train_label

