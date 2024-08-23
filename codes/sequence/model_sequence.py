import torch
import torch.nn as nn
import numpy as np
class recommendation_model(nn.Module):
    def __init__(self, item_num, int_embed_size, geo_embed_size, hidden_size, beta):
        super(recommendation_model, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.item_num = item_num+1
        self.beta = beta
        self.hidden_size=hidden_size
        self.int_embed_size = int_embed_size
        self.geo_embed_size = geo_embed_size
        self.embed_ingoing = nn.Embedding(self.item_num,int(geo_embed_size),padding_idx=self.item_num-1)
        self.embed_outgoing = nn.Embedding(self.item_num,int(geo_embed_size),padding_idx=self.item_num-1)

        self.embed_history = nn.Embedding(self.item_num, int_embed_size,padding_idx=self.item_num-1)
        self.embed_target = nn.Embedding(self.item_num, int_embed_size,padding_idx=self.item_num-1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.loss_func = nn.BCELoss()
        self.softmax = nn.Softmax(dim=-1)

        self.attn_layer1 = nn.Linear(int_embed_size, hidden_size)
        self.attn_layer2 = nn.Linear(hidden_size, 1, bias = False)

        self.attn_layer_geo1 = nn.Linear(geo_embed_size, hidden_size)
        self.attn_layer_geo2 = nn.Linear(hidden_size, 1, bias = False)

        self.drop = nn.Dropout()
        self.cos = nn.CosineSimilarity(dim=-1)
        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.embed_history.weight, std=0.01)
        nn.init.normal_(self.embed_target.weight, std=0.01)
        nn.init.normal_(self.embed_ingoing.weight, std=0.01)
        nn.init.normal_(self.embed_outgoing.weight, std=0.01)
        with torch.no_grad():
            self.embed_history.weight[self.item_num-1] = torch.zeros(self.int_embed_size)
            self.embed_target.weight[self.item_num-1] = torch.zeros(self.int_embed_size)
            self.embed_ingoing.weight[self.item_num-1] = torch.zeros(int(self.geo_embed_size))
            self.embed_outgoing.weight[self.item_num-1] = torch.zeros(int(self.geo_embed_size))
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, history, target, near_pois, ptoc, ctof):
        temp = np.zeros([1,len(near_pois[0])],dtype=np.int64)+(self.item_num-1)
        near_pois = np.append(near_pois,temp ,axis = 0)
        t1 = self.embed_ingoing(torch.LongTensor(near_pois).cuda())
        t2 = self.embed_outgoing(torch.LongTensor(near_pois).cuda())
        a_sus, a_inf = self.self_attention(t1,t2)
        prediction = self.attention_network(history,target, a_sus, a_inf,ptoc, ctof)
    
        return self.sigmoid(prediction)

    def attention_network(self, user_history, target_item, susceptibility, influence, ptoc, ctof):        
        history = self.embed_history(user_history)

        target = self.embed_target(target_item) 

        batch_dim = len(target)
        target = torch.reshape(target,(batch_dim, 1,-1))

        input_item = history * target
        input_ptoc = influence[ptoc] * torch.reshape(susceptibility[target_item], (batch_dim, 1,-1))
        a = susceptibility[ctof]
        b = torch.reshape(influence[target_item], (batch_dim, 1,-1))
        input_ctof = a*b
        input_geo = input_ptoc+input_ctof
        input = self.drop(self.attn_layer1(input_item))
        result1 = self.relu(input) 
        
        result2 = self.attn_layer2(result1)
        
        exp_A = torch.exp(result2)
        exp_A = exp_A.squeeze(dim=-1)
        mask = self.get_mask(user_history,target_item)
        exp_A = exp_A * mask
        exp_sum = torch.sum(exp_A,dim=-1) 
        exp_sum = torch.pow(exp_sum, self.beta)
        
        attn_weights = torch.divide(exp_A.T,exp_sum).T
        attn_weights = attn_weights.reshape([batch_dim,-1, 1])
        result = history * attn_weights
        target = target.reshape([batch_dim,-1,1]) 
        
        prediction = torch.bmm(result, target).squeeze(dim=-1)
        prediction = torch.sum(prediction, dim = -1) 
        
        geo = torch.sum(torch.sum(input_geo,dim=-1),dim=-1)
        return prediction+geo

    def get_mask(self, user_history, target_item):
        target_item = target_item.reshape([len(target_item),1])
        mask = user_history != target_item
        return mask
    def loss_function(self, prediction, label):
        return self.loss_func(prediction, label)
    
    def self_attention(self, ingoing, outgoing):
        q = self.embed_ingoing(torch.LongTensor(np.arange(self.item_num)).to(self.device)).reshape(self.item_num,1,int(self.geo_embed_size))
        # q = ingoing[:,0,:].reshape(self.item_num,1,int(self.geo_embed_size/2))
        k_in = ingoing.reshape(self.item_num,int(self.geo_embed_size),-1)
        v_in = ingoing
        
        t3 = self.softmax(torch.bmm(q,k_in)/torch.sqrt(torch.tensor(self.geo_embed_size)))
        result_in = torch.bmm(t3,v_in).squeeze()
        
        q = self.embed_outgoing(torch.LongTensor(np.arange(self.item_num)).to(self.device)).reshape(self.item_num,1,int(self.geo_embed_size))
        # q = outgoing[:,0,:].reshape(self.item_num,1,int(self.geo_embed_size/2))
        k_out = outgoing.reshape(self.item_num,int(self.geo_embed_size),-1)
        v_out = outgoing
        
        t3 = self.softmax(torch.bmm(q,k_out)/torch.sqrt(torch.tensor(self.geo_embed_size)))
        result_out = torch.bmm(t3,v_out).squeeze()
        return result_in, result_out
 