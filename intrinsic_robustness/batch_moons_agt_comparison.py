""" Code to generate visualisation of AGT vs naive DPA vs naive FA on the halfmoons dataset. """
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
import os
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

# %%
dev = "cuda" if torch.cuda.is_available() else "cpu"
# dev = "cpu"
device = torch.device(dev)
print(f"Using device: {device}")

if not os.path.exists('figures'):
    os.makedirs('figures')

save_dir = '.moons_results_int_rob'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# %%
# get data
ensemble_size = 20
"""Initialise the halfmoons training data."""
seed = 0
train_dataset_size_per_member = 10000  # number of samples per batch
batch_size = 5000
test_size = 1000
# n_batches = 3  # number of batches per epoch
n_epochs = 4  # number of epochs

torch.manual_seed(seed)
# load the dataset
x, y = sklearn.datasets.make_moons(noise=0.1, n_samples=ensemble_size*train_dataset_size_per_member + test_size, random_state=seed)
# to make it easier to train, we'll space the moons out a bit and add some polynomial features
x[y==0, 1] += 0.2
x = np.hstack((x, x**2, (x[:, 0] * x[:, 1])[:, None], x**3))
# perform a test-train split
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x, y, test_size=test_size / (ensemble_size*train_dataset_size_per_member + test_size), random_state=seed
)   

# convert into pytorch dataloaders
x_train, y_train = torch.from_numpy(x_train).float().to(device), torch.from_numpy(y_train).to(device)
x_test, y_test = torch.from_numpy(x_test).float().to(device), torch.from_numpy(y_test).to(device)
dataset_train = torch.utils.data.TensorDataset(x_train, y_train)
dataset_test = torch.utils.data.TensorDataset(x_test, y_test)
# dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batchsize, shuffle=True)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=250, shuffle=False)

# %%
"""Train an ensemble of classifiers on the AGT dataset."""
config = agt.AGTConfig(
    learning_rate=2.0,
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
    model_t = torch.nn.Sequential(torch.nn.Linear(7, 2).to(device))
    bounded_model_t = agt.bounded_models.IntervalBoundedModel(model_t)
    # create a disjoint subset of the training data
    dataset_train_t = torch.utils.data.Subset(dataset_train, indices[t::ensemble_size])  # type: ignore
    dataloader_t = torch.utils.data.DataLoader(dataset_train_t, batch_size=batch_size, shuffle=True)
    ensemble.append((bounded_model_t, dataloader_t))

# %%
gridsize = 100
test_size = gridsize ** 2

# define a grid of points and add features
x0 = torch.linspace(-1.5, 2.5, gridsize)
x1 = torch.linspace(-1.2, 2.2, gridsize)
X0, X1 = torch.meshgrid(x0, x1)
X0f, X1f = X0.flatten(), X1.flatten()
grid_data = torch.stack((X0f, X1f, X0f**2, X1f**2, X0f * X1f, X0f ** 3, X1f ** 3), dim=1)
grid_data = grid_data.to(device)

# %%
# compute b up to N=5
n = len(grid_data)
m = len(ensemble)
b = torch.zeros((m, n)).to(device)
N = 5

trained_ensemble = []
for (i, (bounded_model_t, dataloader_t)) in enumerate(ensemble):
    trained_models = {}
    # Get the largest value of poisoned points before model prediction is no longer robust
    # Train models with different values of k, always compute for k_poison=N
    k_poisons = list(range(1, N+1))
    for k_poison in k_poisons: # try by enumerating all values of k_poison
        model_copy = copy.deepcopy(bounded_model_t)
        print(f"Training model {i} with k={k_poison}")
        torch.manual_seed(seed)
        conf = copy.deepcopy(config)
        conf.k_private = k_poison
        trained_model = agt.privacy_certified_training(model_copy, conf, dataloader_t, dataloader_test)
        trained_models[k_poison] = copy.deepcopy(trained_model)
    
        if k_poison == 1:
            trained_ensemble.append(copy.deepcopy(trained_model))
        
    # tensor of size (1xlen(grid_data))
    bs = agt.privacy_utils.compute_max_certified_k(grid_data, trained_models) + 1
    b[i] = bs.clone()
torch.save(b, f"{save_dir}/moons_agt_bs.pth")



"""Check the corresponding poisoning guarantees."""

# compute the ensemble votes and certificates
scores = torch.zeros((test_size, ensemble_size, 2))
counts = torch.zeros(test_size).to(device)  # will store the vote counts
cert_count = torch.zeros(test_size).to(device)  # will store the number of classifier whose predictions are certified
for i, bounded_model in enumerate(trained_ensemble):
    logits = bounded_model.forward(grid_data)
    scores[:, i, :] = logits
    pred = logits.argmax(dim=1)
    counts += 2 * pred - 1
    cert_preds = agt.test_metrics.certified_predictions(bounded_model, grid_data, return_proportion=False)
    cert_count += cert_preds

# compute the number of votes that must be flipped to cause the ensemble prediction to flip, which is half the distance
# to zero
g = torch.ceil(torch.abs(counts) / 2) - 1
torch.save(g, f"{save_dir}/g.pth")
torch.save(scores, f"{save_dir}/scores.pth")
ensemble_preds = (counts >= 0).float()
torch.save(ensemble_preds, f"{save_dir}/ensemble_preds_agt_dpa.pth")

# %%
""" Solve MILP with Gurobi for AGT"""
k_poison = N
print(f"Running batch certification for AGT with N={k_poison}")
model = gp.Model("Certification")

n = test_size # number of test samples
g = torch.load(f"{save_dir}/g.pth")
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
    }, f"{save_dir}/moons_agt_p_k={k_poison}.pth")

# %%
""" Solve MILP with Gurobi for naive DPA"""
print(f"Running batch certification for naive DPA with N={k_poison}")
model = gp.Model("Certification_DPA")

m = ensemble_size
n = test_size # number of test samples
g = torch.load(f"{save_dir}/g.pth")
gs = g[:n]# number of votes to flip before ensemble prediction flips for each test sample

# DPA b
bs = torch.zeros((m, n)).to(device)
for i in range(m):
    # tensor of size (1xlen(grid_data))
    bs[i] = torch.ones((n))

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

    # compute inner sum (#flipped votes): sum_i(1{p[i] > b[i][k]})
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
        'z': z_values
    }, f"{save_dir}/moons_dpa_p_k={k_poison}.pth")



# %%
"""FA"""
d = 2
k = 20
ensemble_size = d * k
ensemble = []
torch.manual_seed(seed)
indices = torch.randperm(len(dataset_train))
partitions = [t.tolist() for t in list(torch.chunk(indices, ensemble_size))]
shifts = random.sample(range(k), d)
print(shifts)

idxgroup = [[] for _ in range(ensemble_size)]
for i, h in enumerate(partitions):
    for shift in shifts:
        idxgroup[(i + shift)%ensemble_size] += h
        
for t in range(ensemble_size):
    trained_models = []
    model_t = torch.nn.Sequential(torch.nn.Linear(7, 2).to(device))
    bounded_model_t = agt.bounded_models.IntervalBoundedModel(model_t)
    # create a disjoint subset of the training data
    dataset_train_t = torch.utils.data.Subset(dataset_train, idxgroup[t])  # type: ignore
    dataloader_t = torch.utils.data.DataLoader(dataset_train_t, batch_size=batch_size, shuffle=True)
    ensemble.append((bounded_model_t, dataloader_t))

""" Train FA ensemble"""
n = len(grid_data)
m = len(ensemble)
b = torch.zeros((m, n)).to(device)

# compute the ensemble votes and certificates
scores = torch.zeros((test_size, ensemble_size, 2))
counts = torch.zeros(test_size).to(device)  # will store the vote counts
cert_count = torch.zeros(test_size).to(device)  # will store the number of classifier whose predictions are certified

torch.manual_seed(seed)
trained_ensemble = []
for (i, (bounded_model_t, dataloader_t)) in enumerate(ensemble):
    # Get the largest value of poisoned points before model prediction is no longer robust
    # Train models with different values of k, always compute for k_poison=N
    model_copy = copy.deepcopy(bounded_model_t)
    print(f"Training model {i} with k={k_poison}")
    conf = copy.deepcopy(config)
    conf.k_poison = 1
    trained_model = agt.poison_certified_training(model_copy, conf, dataloader_t, dataloader_test)
    
    """Check the corresponding poisoning guarantees."""
    logits = trained_model.forward(grid_data)
    scores[:, i, :] = logits
    pred = logits.argmax(dim=1)
    counts += 2 * pred - 1
ensemble_preds = (counts >= 0).float()

# compute the number of votes that must be flipped to cause the ensemble prediction to flip, which is half the distance
# to zero
g = torch.ceil(torch.abs(counts) / 2) - 1
torch.save(g, f"{save_dir}/g_fa.pth")
torch.save(scores, f"{save_dir}/scores_fa.pth")
torch.save(ensemble_preds, f"{save_dir}/ensemble_preds_fa.pth")

""" Solve MILP with Gurobi for FA"""
print(f"Running batch certification for FA with N={k_poison}")
model = gp.Model("Certification FA")

m = ensemble_size
n = test_size # number of test samples
g = torch.load(f"{save_dir}/g_fa.pth")
gs = g[:n]# number of votes to flip before ensemble prediction flips for each test sample

# Naive FA b
bs = torch.zeros((m, n)).to(device)
for i in range(m):
    # tensor of size (1xlen(grid_data))
    bs[i] = torch.ones((n))

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

    # compute inner sum (#flipped votes): sum_i(1{p[i] > b[i][k]})
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
        'z': z_values
    }, f"{save_dir}/moons_fa_p_k={k_poison}.pth")


# %%
def plot_moons(ensemble_preds, uncert, fig_fname):
    fig, ax = plt.subplots()
    ax.set_box_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    preds = ensemble_preds.reshape(gridsize, gridsize).cpu().numpy()
    ax.contour(X0, X1, preds, levels=[0.5])
    cert_preds_0 = ((ensemble_preds == 0) & (uncert == 0)).float()  # where the model always predicts class 0
    cert_preds_1 = ((ensemble_preds == 1) & (uncert == 0)).float()  # where the model always predicts class 1
    cert_preds = (cert_preds_0 - cert_preds_1).reshape(gridsize, gridsize).cpu().numpy()
    print((cert_preds == 0).sum())
    pastel_red = "#FE2C02"   # Soft pinkish-red
    pastel_blue = "#015FAA"  # Light pastel blue
    cmap = ListedColormap([pastel_red, "white", pastel_blue])
    ax.contourf(X0, X1, cert_preds, cmap=cmap, levels=[-1.5, -0.5, 0.0, 0.5, 1.5], alpha=0.5)

    # plot the moons
    ax.scatter(x_test.cpu()[y_test.cpu() == 0, 0], x_test.cpu()[y_test.cpu() == 0, 1], s=25, edgecolors="k", color="blue")
    ax.scatter(x_test.cpu()[y_test.cpu() == 1, 0], x_test.cpu()[y_test.cpu() == 1, 1], s=25, edgecolors="k", color="red")

    # save fig
    plt.savefig(fig_fname, dpi=300, bbox_inches='tight')

# %%
""" Visualise the results (AGT) """
ensemble_preds = torch.load(f"{save_dir}/ensemble_preds_agt_dpa.pth")
agt_uncert = torch.tensor(torch.load(f"{save_dir}/moons_agt_p_k={N}.pth")['z']).to(device)
plot_moons(ensemble_preds, agt_uncert, f"figures/agt_moons_N={N}.pdf")

# %%
""" Visualise the results (DPA) """
ensemble_preds = torch.load(f"{save_dir}/ensemble_preds_agt_dpa.pth")
dpa_uncert = torch.tensor(torch.load(f"{save_dir}/moons_dpa_p_k={N}.pth")['z']).to(device)
plot_moons(ensemble_preds, dpa_uncert, f"figures/dpa_moons_N={N}.pdf")

# %%
""" Visualise the results (FA) """
ensemble_preds = torch.load(f"{save_dir}/ensemble_preds_fa.pth")
fa_uncert = torch.tensor(torch.load(f"{save_dir}/moons_fa_p_k={N}.pth")['z']).to(device)
plot_moons(ensemble_preds, fa_uncert, f"figures/fa_moons_N={N}.pdf")