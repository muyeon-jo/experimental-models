import random
import numpy as np
import torch

def train_batch(train_positive, num_poi, uid, negative_num):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    item_list = np.arange(num_poi).tolist()

    positives = train_positive[uid]
    histories = np.array([positives])

    ptoc = np.zeros(len(positives)).reshape(1,-1)+(num_poi)
    ctof = np.array(positives,dtype=np.int64).reshape(1,-1)

    ctof[0][0]=num_poi

    for i in range(1,len(positives)):
        
        ptoc = np.append(ptoc,ptoc[i-1].reshape(1,-1),axis = 0)
        ptoc[i][i-1] = positives[i-1]

        
        ctof = np.append(ctof,ctof[i-1].reshape(1,-1),axis = 0)
        ctof[i][i]=num_poi
    a=[]
    b=[]
    for i in range(negative_num+1):
        a.append(ptoc)
        b.append(ctof)
    ptoc = np.stack(a,axis=1).reshape(-1,len(train_positive[uid]))
    ctof = np.stack(b,axis=1).reshape(-1,len(train_positive[uid]))

    # histories = np.zeros(len(train_positive[uid])).reshape(1,-1)+(num_poi)
    # histories[0][0] = train_positive[uid][0]
    # for i in range(1,len(train_positive[uid])-1):
    #     histories = np.append(histories,histories[i-1].reshape(1,-1),axis = 0)
    #     histories[i][i] = train_positive[uid][i]
    # a=[]
    # for i in range(negative_num+1):
    #     a.append(histories)
    # temp = np.stack(a,axis=1).reshape(-1,len(train_positive[uid]))
    # time_idx = np.repeat(np.arange(len(positives)),negative_num)
    
    #네거티브 
    # b=[]
    # for i in positives:
    #     tt = set(item_list)
    #     tt.discard(i)
    #     negative = list(tt)
    #     nidx = np.random.randint(0,len(negative),size=negative_num)
    #     b.append(np.array(negative)[nidx])
    # negatives = np.stack(b,axis=0).reshape([-1,negative_num])
    
    negative = list(set(item_list)-set(train_positive[uid]))
    random.shuffle(negative)
    nidx = np.random.randint(0,len(negative),size=len(train_positive[uid])*negative_num)
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
    ptoc = torch.LongTensor(ptoc).to(DEVICE)
    ctof = torch.LongTensor(ctof).to(DEVICE)
    return user_history, train_data, train_label, ptoc, ctof

def test_batch(train_positive, uid, item_len):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    history = train_positive[uid]
    negative = list(set(range(item_len))-set(history))
    # histories = np.array([history]).repeat(len(negative),axis=0)
    histories = np.array([history])

    
    ptoc = np.array(history,dtype=np.int64).reshape(1,-1)
    ctof = np.zeros(len(history)).reshape(1,-1)+(item_len) 

    ctof[0][0]=item_len
    a=[]
    b=[]
    for i in range(len(negative)):
        a.append(ptoc)
        b.append(ctof)
    ptoc = np.stack(a,axis=1).reshape(-1,len(train_positive[uid]))
    ctof = np.stack(b,axis=1).reshape(-1,len(train_positive[uid]))
    negative_label = np.array([0]).repeat(len(negative))

    user_history = histories
    train_data = negative
    train_label = torch.tensor(negative_label,dtype=torch.float32).to(DEVICE)

    user_history=torch.LongTensor(user_history).to(DEVICE)
    train_data=torch.LongTensor(train_data).to(DEVICE)
    ptoc = torch.LongTensor(ptoc).to(DEVICE)
    ctof = torch.LongTensor(ctof).to(DEVICE)
    return user_history, train_data, train_label, ptoc, ctof

