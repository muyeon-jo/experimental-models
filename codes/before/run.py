from datetime import datetime
from argparse import ArgumentParser
import os
import datasets
from batches import *
from model import *
import time
import random
import validation as val
import torch.cuda
import torch
from save import save_experiment_result

def train(train_matrix, test_positive, val_positive, dataset,args):

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
    model = recommendation_model(num_items, args.int_embed_size, args.geo_embed_size,args.hidden_dim,0.3).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lamda)
    for e in range(args.epochs):
        model.train()
        train_loss = 0.0
        start_time = int(time.time())

        idx = list(range(num_users))
        random.shuffle(idx)
        for buid in idx:
            optimizer.zero_grad()
            user_history , train_data, train_label= train_batch(train_matrix, num_items, buid, args.num_ng)
            temp = user_history.expand([len(train_label),len(user_history[0])])
            prediction = model(temp, train_data, dataset.nearPOI)

            # w = 5
            # m = 0.9
            # hist_num = train_label.sum()
            # neg_num = len(train_label)-hist_num
            # tt = 1-prediction
            # pos = (train_label*tt).sum()
            # ttt = model.relu(prediction-m)
            # neg = ((1-train_label)*ttt).sum() * w /neg_num
            # loss = pos+neg

            loss = model.loss_func(prediction,train_label)
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
                # model.BSPM()
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
if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = ArgumentParser(description="model args")
    parser.add_argument('-d','--dataset', type=int, default=3, help='1: Tokyo, 2: NewYork, 3: Philadelphia')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('-lr', '--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-l','--lamda', type=float, default=1e-7, help='regularization weight')
    parser.add_argument('-ng', '--num_ng', type=int, default=4, help='number of negative sample')
    parser.add_argument('-hi','--hidden_dim', type=int, default=128, help='hidden_dimmension')
    parser.add_argument('-n','--near_num', type=int, default=150, help='number of closely located POI')
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