import torch
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import argparse
import os
import batch_certify_utils

parser = argparse.ArgumentParser(description='FA MILP Certification')
parser.add_argument('--evaluations',  type=str, help='name of evaluations directory')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
parser.add_argument('--dataset', type=str, default="cifar", help='cifar or mnist')

parser.add_argument('--k', default = 50, type=int, help='the inverse of sensitivity')
parser.add_argument('--d', default = 1, type=int, help='number of duplicates per sample')

parser.add_argument('--batch_size', type=int, default=100, help='Test batch size for certification')
parser.add_argument('--from_idx', type=int, default=0, help='Index of test set to start certification from')
parser.add_argument('--num_batches', type=int, default=100, help='Number of batches to run certification on')
parser.add_argument('--k_poisons', type=int, nargs='+', help='Number of points the adversary can poison')

args = parser.parse_args()
if not os.path.exists('./batch_certs'):
    os.makedirs('./batch_certs')

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

filein = torch.load('./evaluations/'+args.evaluations + '.pth', map_location=device)
ensemble_size = filein['scores'].shape[1]
assert ensemble_size == args.k*args.d, f"ensemble size should be equal to k*d {args.k*args.d}, but got {ensemble_size} "
total_test_size = filein['scores'].shape[0]
print("total test size", total_test_size)
print("ensemble size", ensemble_size)

if not os.path.exists('./batch_certs/fa_roe_' + str(args.evaluations)):
    os.makedirs('./batch_certs/fa_roe_' + str(args.evaluations))
    
partition_file = torch.load(f"./train/FiniteAggregation_hash_mean_{args.dataset}_k{args.k}_d{args.d}.pth", weights_only=False, map_location=device)
idxgroup = partition_file['idx']
shifts = partition_file['shifts']
assert len(shifts) == args.d, f"shifts should be of length {args.d}, but got {len(shifts)}"
# get length of each element in idxgroup as batchsize
batchsizes = [len(idxgroup[i]) for i in range(len(idxgroup))]

def get_realised_ps(p_values, shifts, ensemble_size):
    """
    Returns:
        list: list of realised p values that affects the dataset that we use to train in FA
    """
    # find which members contain data point from which partition
    h_spread_inv = [[] for i in range(ensemble_size)]
    for i in range(ensemble_size):
        for shift in shifts:
            h_spread_inv[i].append((i - shift) % ensemble_size)
            
    p_realised = []
    for indices in h_spread_inv:
        expr = sum(p_values[j] for j in indices)  # creates a symbolic Gurobi linear expression
        p_realised.append(expr)
        
    return p_realised

def certify_batch_fa_roe(k_poison, pred_classes, labels, per_datapoint_acc, shifts, from_idx, to_idx):
    """ Solve MILP with Gurobi """
    model = gp.Model("Certification")
    
    # Use DPA+ROE certificates here, we consider the FA certificates in the objective function
    gs = batch_certify_utils.compute_g_roe("v_dpa_roe_" + str(args.evaluations), from_idx, to_idx) # number of votes to flip before ensemble prediction flips for each test sample
    bs = batch_certify_utils.compute_b_naive(k_poison, per_datapoint_acc, ensemble_size) # number of datapoints to poison for each member before prediction for each test sample changes
    n = len(labels) # number of test samples
    print(f"Gs: {gs}")
    print(f"Bs: {bs}")

    # Define variables
    # Relaxing p to continuous for faster solving (gives the same result)
    p = model.addVars(ensemble_size, vtype=GRB.CONTINUOUS, lb=0, name="poisoning_vector") # poisoning vector that should sum up to N
    z = model.addVars(n, vtype=GRB.BINARY, name="pred_flipped_indicator")
    # Total points poisoned for each member after FA
    p_realised = get_realised_ps(p, shifts, ensemble_size)

    # Outer loop, compute outer sum: sum_k(1{g_k <= sum_i(1{p[i] > b[i][k]})})
    for k in range(n):
        # Create decision variables
        z_k = model.addVars(ensemble_size, vtype=GRB.BINARY, name=f"z_{k}")  # Binary indicator variables for {p_i > b_ik}

        # compute inner sum (#flipped votes): sum_i(1{p[i] > b[i][k]})
        for i in range(ensemble_size):
            model.addGenConstrIndicator(z_k[i], 1, p_realised[i] - bs[i][k], GRB.GREATER_EQUAL, 0, name=f"vote_flipped_indicator_{i}{k}")

        num_flipped_votes = gp.quicksum(z_k[i] for i in range(ensemble_size))
        model.addGenConstrIndicator(z[k], 1, gs[k]-num_flipped_votes+1, GRB.LESS_EQUAL, 0, name=f"pred_flipped_indicator_{k}")

    num_flipped_preds = gp.quicksum(z[i] for i in range(n))

    # Define objective function
    model.setObjective((1/n)*num_flipped_preds, GRB.MAXIMIZE)

    # Constraint: #total poisoned points == N
    model.addConstr(gp.quicksum(p[i] for i in range(ensemble_size)) == k_poison)
    
    # Constraint: # poisoned points for each member <= batchsize
    for i in range(ensemble_size):
        model.addConstr(p[i] <= batchsizes[i])
        
    model.setParam('TimeLimit', 1800) # 30 minutes
    
    # loosen optimality tolerance
    # model.setParam('MIPGap', 1e-2)  # Allow 1% gap
    
    model.update()

    # Optimize
    model.optimize()

    # Print results
    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        vars = {var.VarName: var.X for var in model.getVars()}

        # Extract p and z values by matching their variable names
        p_values = [vars[f'poisoning_vector[{i}]'] for i in range(ensemble_size)]
        z_values = [vars[f'pred_flipped_indicator[{k}]'] for k in range(n)]
        worst_case_accuracy = 1-model.objVal
        opt_gap = model.MIPGap
        if model.status == GRB.TIME_LIMIT:
            print("Gurobi reached time limit, returning dual solution found.")
            worst_case_accuracy = 1 - model.ObjBound
        print("Worst case flipped:", p_values)
        print("Worst case accuracy", worst_case_accuracy)
        print(f"Solve time: {model.Runtime:.4f} seconds")
        return worst_case_accuracy, p_values, z_values, opt_gap
    
    print("MILP cannot be solved.")
    return None

def run_batch(from_idx, to_idx, shifts):
    pred_classes = torch.argmax(filein['scores'], dim=2)[from_idx:to_idx]
    labels = filein['labels'][from_idx:to_idx]

    for k_poison in args.k_poisons:
        per_datapoint_acc, acc = batch_certify_utils.find_nominal_accuracy_and_preds(pred_classes, labels)
        print(f"Poisoning {k_poison} points")
        worst_case_accuracy, p_values, z_values, opt_gap = certify_batch_fa_roe(k_poison, pred_classes, labels, per_datapoint_acc, shifts, from_idx, to_idx)
        cert_accs_milp = (worst_case_accuracy, opt_gap)

        fname = f"batch_certs/fa_roe_{str(args.evaluations)}/cert_accs_N={k_poison}_batch_{from_idx}_{to_idx}.pth"
        torch.save(cert_accs_milp, fname)
        print(f"Results saved to {fname}!")

""" Run in batches """
run_batch_size = args.batch_size
test_idx_end = args.from_idx + args.num_batches * run_batch_size
for i in range(args.from_idx, test_idx_end, run_batch_size):
    print(f"Running batch {i} to {i+run_batch_size}")
    run_batch(i, i+run_batch_size, shifts)
