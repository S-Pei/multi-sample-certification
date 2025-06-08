""" Code to run AGT with varying ensemble sizes. """
# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import abstract_gradient_training as agt
import sklearn
import copy
import gurobipy as gp
from gurobipy import GRB
import json
from tqdm import tqdm
import random
import matplotlib.colors as mcolors
import os

# %%
dev = "cuda" if torch.cuda.is_available() else "cpu"
# dev = "cpu"  # force CPU for this example
device = torch.device(dev)
print(f"Using device: {device}")

save_dir = f".graph_results/ens_run_50k_total_fixed"  # directory to save results

# create the directory if it does not exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

sizes = [1] + list(range(5, 101, 5))  # ensemble sizes to test
# sizes = list(range(75, 101, 5))
for ensemble_size in sizes:
# ensemble_size = 5
    print(f"TRAINING WITH ENSEMBLE_SIZE: {ensemble_size}")
    """Initialise the halfmoons training data."""
    seed = 0
    total_dataset_size = 50000
    train_dataset_size_per_member = total_dataset_size // ensemble_size
    batch_size = train_dataset_size_per_member
    
    # train_dataset_size_per_member = 10000
    # total_dataset_size = train_dataset_size_per_member * ensemble_size
    # batch_size = train_dataset_size_per_member
    test_size = 1000
    # n_batches = 3  # number of batches per epoch
    n_epochs = 4  # number of epochs

    torch.manual_seed(seed)
    # load the dataset
    x, y = sklearn.datasets.make_moons(noise=0.1, n_samples=total_dataset_size + test_size, random_state=seed)
    # to make it easier to train, we'll space the moons out a bit and add some polynomial features
    x[y==0, 1] += 0.2
    x = np.hstack((x, x**2, (x[:, 0] * x[:, 1])[:, None], x**3))
    # perform a test-train split
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=test_size, random_state=seed
    )   

    # convert into pytorch dataloaders
    x_train, y_train = torch.from_numpy(x_train).float().to(device), torch.from_numpy(y_train).to(device)
    x_test, y_test = torch.from_numpy(x_test).float().to(device), torch.from_numpy(y_test).to(device)
    dataset_train = torch.utils.data.TensorDataset(x_train, y_train)
    dataset_test = torch.utils.data.TensorDataset(x_test, y_test)
    # dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batchsize, shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=250, shuffle=False)

    """Train an ensemble of classifiers on the AGT dataset."""
    NOMINAL_CONFIG = agt.AGTConfig(
        learning_rate=3.0,
        n_epochs=4,
        device=dev,
        loss="cross_entropy",
        lr_decay=0.6,
        lr_min=1e-3,
        log_level="DEBUG",
        # paired_poison=True,
        clip_gamma=0.1,
    )

    ensemble = []
    torch.manual_seed(seed)
    indices = torch.randperm(len(dataset_train))
    for t in range(ensemble_size):
        trained_models = []
        model_t = torch.nn.Sequential(
            torch.nn.Linear(7, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2),
        ).to(device)
        bounded_model_t = agt.bounded_models.IntervalBoundedModel(model_t)
        # create a disjoint subset of the training data
        dataset_train_t = torch.utils.data.Subset(dataset_train, indices[t::ensemble_size])  # type: ignore
        dataloader_t = torch.utils.data.DataLoader(dataset_train_t, batch_size=batch_size, shuffle=True)
        ensemble.append((bounded_model_t, dataloader_t))
        
    # compute b
    n = test_size
    m = len(ensemble)
    b = torch.zeros((m, n)).to(device)
    N = 20
    s = 5

    trained_ensemble = []
    for (i, (bounded_model_t, dataloader_t)) in enumerate(ensemble):
        trained_models = {}
        # Get the largest value of poisoned points before model prediction is no longer robust
        # Train models with different values of k, always compute for k_poison=N
        k_poisons = list(range(1, N, s)) + [N]  # include N as the last value
    
        for k_poison in k_poisons: # try by enumerating all values of k_poison
            model_copy = copy.deepcopy(bounded_model_t)
            print(f"Training model {i} with k={k_poison}")
            torch.manual_seed(seed)
            conf = copy.deepcopy(NOMINAL_CONFIG)
            conf.k_private = k_poison
            # conf.epsilon = 0.01
            trained_model = agt.privacy_certified_training(model_copy, conf, dataloader_t, dataloader_test)
            trained_models[k_poison] = copy.deepcopy(trained_model)
        
            if k_poison == 1:
                trained_ensemble.append(copy.deepcopy(trained_model))
            
        # tensor of size (1xlen(grid_data))
        print(len(x_test))
        bs = agt.privacy_utils.compute_max_certified_k(x_test, trained_models) + 1
        b[i] = bs.clone()
        # print(bs)
    

    """Check the corresponding poisoning guarantees."""

    # compute the ensemble votes and certificates
    scores = torch.zeros((test_size, ensemble_size, 2))
    counts = torch.zeros(test_size).to(device)  # will store the vote counts
    for i, bounded_model in enumerate(trained_ensemble):
        logits = bounded_model.forward(x_test)
        scores[:, i, :] = logits
        pred = logits.argmax(dim=1)
        counts += 2 * pred - 1
        b[i, pred != y_test] = N + 1  # set b to N+1 for samples where the prediction is wrong

    torch.save(b, f"{save_dir}/moons_agt_bs_{N}_{ensemble_size}_{s}.pth")
    # compute the number of votes that must be flipped to cause the ensemble prediction to flip, which is half the distance
    # to zero
    g = torch.ceil(torch.abs(counts) / 2) - 1
    ensemble_preds = (counts >= 0).float()
    g[ensemble_preds != y_test] = -1
    nominal_acc = (ensemble_preds == y_test).float().mean().item()
    print(f"Nominal accuracy: {nominal_acc:.4f}")
    torch.save(g, f"{save_dir}/g_{N}_{ensemble_size}_{s}.pth")
    torch.save(scores, f"{save_dir}/scores_{N}_{ensemble_size}_{s}.pth")
    torch.save({
        "preds": ensemble_preds,
        "nom_acc": nominal_acc
        }, 
    f"{save_dir}/ensemble_preds_agt_dpa_{N}_{ensemble_size}_{s}.pth")

    """ Solve MILP with Gurobi """
    k_poison = N
    print(k_poison)
    model = gp.Model("Certification")

    n = test_size # number of test samples
    gs = g[:n]# number of votes to flip before ensemble prediction flips for each test sample
    bs = b[:, :n] # number of datapoints to poison for each member before prediction for each test sample changes

    print(f"Gs: {gs}")
    print(f"Bs: {bs}")

    # Define variables
    # Relaxing p to continuous for faster solving (gives the same result)
    p = model.addVars(ensemble_size, vtype=GRB.CONTINUOUS, lb=0, name="poisoning_vector") # poisoning vector that should sum up to N
    z = model.addVars(n, vtype=GRB.BINARY, name="pred_flipped_indicator")

    # Outer loop, compute outer sum: sum_k(1{g_k <= sum_i(1{p[i] > b[i][k]})})
    for k in range(n):
        # Create decision variables
        z_k = model.addVars(ensemble_size, vtype=GRB.BINARY, name=f"z_{k}")  # Binary indicator variables for {p_i > b_ik}

        # compute inner sum (#flipped votes): sum_i(1{p[i] >= b[i][k]})
        for i in range(ensemble_size):
            model.addGenConstrIndicator(z_k[i], 1, p[i] - bs[i][k], GRB.GREATER_EQUAL, 0, name=f"vote_flipped_indicator_{i}{k}")

        num_flipped_votes = gp.quicksum(z_k[i] for i in range(ensemble_size))
        model.addGenConstrIndicator(z[k], 1, gs[k]-num_flipped_votes+1, GRB.LESS_EQUAL, 0, name=f"pred_flipped_indicator_{k}")

    num_flipped_preds = gp.quicksum(z[i] for i in range(n))

    # Define objective function
    model.setObjective((1/n)*num_flipped_preds, GRB.MAXIMIZE)

    # Constraint: #total poisoned points == N
    model.addConstr(gp.quicksum(p[i] for i in range(ensemble_size)) == k_poison)
        
    model.setParam('TimeLimit', 1800) # 15 minutes

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
        torch.save({
            'cert_acc': worst_case_accuracy,
            'p': p_values,
            'z': z_values,
        }, f"{save_dir}/moons_agt_p_k={k_poison}_{ensemble_size}_{s}.pth")