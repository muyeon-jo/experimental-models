from datetime import datetime
from argparse import ArgumentParser
import os
import datasets as datasets
from batches import *
from model_dataloader import *
import time
import random
import validation_dataloader as val
import torch.cuda
import torch
from save import save_experiment_result
import dataloader
from torch.utils.data import DataLoader
# import evaluation_BSPM
# from test_model import BSPM, convert_sp_mat_to_sp_tensor
def train(train_matrix, test_positive, val_positive, dataset,args):
    batch_size = 64
    now = datetime.now()
    model_directory = "./model/"+now.strftime('%Y-%m-%d %H_%M_%S')
    result_directory = "./result/"+now.strftime('%Y-%m-%d %H_%M_%S')
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    max_recall = 0.0
    k_list=[5, 10, 15, 20, 25, 30]

    with open(result_directory+"/setting.txt","w") as setting_f:
        setting_f.write("lr:{}\n".format(str(args.lr)))
        setting_f.write("lamda:{}\n".format(str(args.lamda)))
        setting_f.write("epochs:{}\n".format(str(args.epochs)))
        setting_f.write("geo_dim:{}\n".format(str(args.geo_embed_size)))
        setting_f.write("int_dim:{}\n".format(str(args.int_embed_size)))
        setting_f.write("hidden_dim:{}\n".format(str(args.hidden_dim)))
        setting_f.write("num_ng:{}\n".format(str(args.num_ng)))
        setting_f.write("dataset:{}\n".format(str(dataset.directory_path)))

    num_users = dataset.user_num
    num_items = dataset.poi_num
    model = recommendation_model(num_items, args.int_embed_size, args.geo_embed_size,args.hidden_dim,0.5).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lamda)
    dat = dataloader.mydataset(train_matrix,num_items,num_users,args.num_ng)
    dataload = DataLoader(dat,collate_fn=dat.my_collate_fn,batch_size=batch_size,shuffle=True,num_workers=6)
    for e in range(args.epochs):
        model.train()
        train_loss = 0.0
        start_time = int(time.time())

        idx = list(range(num_users))
        random.shuffle(idx)
        
        for user_history, train_data, train_label in dataload:
        # for buid in idx:
            optimizer.zero_grad()
            # user_history , train_data, train_label= train_batch(train_matrix, num_items, buid, args.num_ng)
            prediction = model(user_history.to(DEVICE), train_data.to(DEVICE), dataset.nearPOI)
            loss = model.loss_func(prediction,train_label.to(DEVICE))
            train_loss += loss.item()
            if loss.item() < 0.0:
                print("a")
            loss.backward() 
            optimizer.step() 
        end_time = int(time.time())
        print("Train Epoch: {}; time: {} sec; loss: {:.4f}".format(e+1, end_time-start_time,train_loss))
        if (e+1)%5 == 0:
            model.eval() 
            with torch.no_grad():
                start_time = int(time.time())
                precision_v, recall_v, hit_v, precision_t, recall_t, hit_t = val.validation(model,num_users,test_positive,val_positive,train_matrix,k_list,dataset.nearPOI)
                if(max_recall < recall_v[1]):
                    max_recall = recall_v[1]
                    torch.save(model, model_directory+"/model")
                    save_experiment_result(result_directory,[recall_t,precision_t,hit_t],k_list,e+1)
                end_time = int(time.time())
                print("eval time: {} sec".format(end_time-start_time))

def run(dataset,arg):
    train_matrix, test_positive, val_positive, place_coords = dataset.generate_data(0,args.near_num)
    print("train data generated")
    
    print("train start")
    train(train_matrix, test_positive, val_positive, dataset,arg)

# def run_BSPM(dataset,args):
#     train_matrix, test_positive, val_positive, place_coords = dataset.generate_data(0,args.near_num)
#     now = datetime.now()
#     k_list=[5, 10, 15, 20, 25, 30]
#     adj_mat = dataset.user_POI_Graph
#     lm = BSPM(adj_mat)
#     lm.train()

#     batch_ratings = convert_sp_mat_to_sp_tensor(adj_mat)
#     rating = lm.getUsersRating(batch_ratings)
#     result_directory = "./result/"+now.strftime('%Y-%m-%d %H_%M_%S')
#     if not os.path.exists(result_directory):
#         os.makedirs(result_directory)
#     with open(result_directory+"/setting.txt","w") as setting_f:
#         setting_f.write("lr:{}\n".format(str(args.lr)))
#         setting_f.write("lamda:{}\n".format(str(args.lamda)))
#         setting_f.write("epochs:{}\n".format(str(args.epochs)))
#         setting_f.write("geo_dim:{}\n".format(str(args.geo_embed_size)))
#         setting_f.write("int_dim:{}\n".format(str(args.int_embed_size)))
#         setting_f.write("hidden_dim:{}\n".format(str(args.hidden_dim)))
#         setting_f.write("num_ng:{}\n".format(str(args.num_ng)))
#         setting_f.write("dataset:{}\n".format(str(dataset.directory_path)))
#     precision_v, recall_v, hit_v, precision_t, recall_t, hit_t = evaluation_BSPM.evaluation(rating,dataset.user_num,val_positive,test_positive,train_matrix)
#     save_experiment_result(result_directory,[recall_t,precision_t,hit_t],k_list,0)
if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = ArgumentParser(description="model args")
    parser.add_argument('-d','--dataset', type=int, default=1, help='1: Tokyo, 2: NewYork, 3: Philadelphia')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('-lr', '--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-l','--lamda', type=float, default=1e-8, help='regularization weight')
    parser.add_argument('-ng', '--num_ng', type=int, default=50, help='number of negative sample')
    parser.add_argument('-hi','--hidden_dim', type=int, default=128, help='hidden_dimmension')
    parser.add_argument('-n','--near_num', type=int, default=100, help='number of closely located POI')
    parser.add_argument('-i','--int_embed_size', type=int, default=64, help='dimmension of inherent embedding')
    parser.add_argument('-g','--geo_embed_size', type=int, default=64, help='dimmension of geographical embedding')
    args = parser.parse_args()
    if args.dataset == 1:
        dataset_ = datasets.Tokyo()
    elif args.dataset == 2:
        dataset_ = datasets.NewYork()
    elif args.dataset == 3:
        dataset_ = datasets.Philadelphia()
    run(dataset_,args)