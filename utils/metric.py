import torch 
import sklearn

def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return acc

def micro_f1(preds, labels):
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=list(range(1, 44))) * 100.0