import torch
import numpy as np
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_nominal_accuracy_and_preds(y_preds, labels, num_classes=10):
    """Compute unpoisoned predictions"""
    votes = [np.bincount(pred_class.to(torch.int64).cpu().numpy(), minlength=num_classes) for pred_class in y_preds]
    preds = np.argmax(votes, axis=1)
    nominal_accuracy = y_preds == labels.unsqueeze(1)
    nominal_accuracy = nominal_accuracy.T
    accuracy = np.mean(preds == labels.cpu().numpy())
    print(f"Nominal Accuracy: {accuracy:.4f}")

    return nominal_accuracy, accuracy

def compute_g_naive(y_preds, y_test, num_classes):
    """ 
    Computes the number of votes needed to change the prediction which is given by

        (max(votes) - max({votes \ argmax(votes)})) / 2
    
    if nominal prediction is incorrect, then g[i] = -1
    
    Args:
    y_preds (torch.Tensor): nominal predictions for each member
    y_test (torch.Tensor): test labels

    Returns:
        torch.Tensor: number of votes needed to change the prediction for each test datapoint
    """
    # calculate votes for each class
    g = torch.zeros(len(y_test), dtype=torch.int64).to(device)

    for k in range(len(y_test)):
        votes = np.bincount(y_preds[k].to(torch.int64).cpu().numpy(), minlength=num_classes) # classes 0 to n-1
        pred = np.argmax(votes)
        # increment votes by 1 if class index is less than preds
        votes[:pred] += 1
        # Sort counts in descending order
        arg_sorted_votes = np.argsort(votes)
        sorted_votes = votes[arg_sorted_votes]
        # print(sorted_votes)

        max_votes = sorted_votes[-1]
        second_max_votes = sorted_votes[-2]

        pred = arg_sorted_votes[-1]
        next_pred = arg_sorted_votes[-2]
        
        if pred == y_test[k]:
            to_flip = (max_votes - second_max_votes)/2
            g[k] = np.floor(to_flip).astype(int)
        else:
            g[k] = -1

    return g

def compute_g_roe(fname, from_idx, to_idx):
    certs = torch.load('./certs/' + fname + '.pth', map_location=device, weights_only=False)
    
    return certs[from_idx:to_idx]
    
def compute_b_naive(N, per_datapoint_acc, ensemble_size):
    """
    Returns:
        tuple[torch.Tensor, torch.Tensor]:
        b (torch.Tensor): matrix of number of datapoints to poison before prediction is no longer robust where

            b[i] is a vector corresponding to member i of the ensemble
            b[i][k] = N + 1 if nominal prediction != y_test[k]
            b[i][k] = 1 otherwise **
    """
    m = ensemble_size
    n = per_datapoint_acc.shape[1]
    b = torch.zeros((m, n)).to(device)
    for i in range(m):
        # tensor of size (1xlen(x_test))
        bs = torch.ones((n))
        bs[~per_datapoint_acc[i]] = N+1
        b[i] = bs.clone()
    # print(bs)
    return b