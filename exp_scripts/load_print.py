""" Loads and prints results from the batch vs pointwise certification comparison """
import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Batch vs Pointwise Certification')
parser.add_argument('--evaluations',  type=str, help='name of evaluations directory')
parser.add_argument('--method',  type=str, help='method to compare: dpa, dpa_roe, fa, fa_roe')
parser.add_argument('--k_poisons', type=int, nargs='+', help='Number of points the adversary can poison')
parser.add_argument('--k', default = 50, type=int, help='the inverse of sensitivity')
parser.add_argument('--d', default = 1, type=int, help='number of duplicates per sample')

args = parser.parse_args()

k_poisons = args.k_poisons
for k_poison in k_poisons:
    res = torch.load(f"results/r_{str(args.method)}_{str(args.evaluations)}_N={k_poison}.pth", weights_only=False)
    print("==========================================")
    print(f"{args.method} Results for k_poison={k_poison}, k={args.k}, d={args.d}:")
    print(f"Batch cert acc: {res['batch_cert_acc']}, pointwise cert acc: {res['pointwise_cert']}, improvement: {res['improvement']}")
    print("==========================================")