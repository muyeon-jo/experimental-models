from batches import *
import torch.cuda
import eval_metrics
import math

def evaluation(matrix, num_users,val_positive,test_positive,train_matrix):
    recommended_list = []
    k_list=[5, 10, 15, 20, 25, 30]
    for user_id in range(num_users):
        pred = matrix[user_id]
        for i in train_matrix.getrow(user_id).indices:
            pred[i] = -1.0
        # pred = torch.tensor(pred).to(DEVICE)
        _, indices = torch.topk(pred, 50)
        recommended_list.append([i.item() for i in indices])
    # torch.cuda.empty_cache()
    precision_v, recall_v, hit_v = eval_metrics.evaluate_mp(val_positive,recommended_list,k_list)
    precision_t, recall_t, hit_t = eval_metrics.evaluate_mp(test_positive,recommended_list,k_list)
    return precision_v, recall_v, hit_v, precision_t, recall_t, hit_t
