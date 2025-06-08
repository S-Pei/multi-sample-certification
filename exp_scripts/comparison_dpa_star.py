""" Script to compare between batch and pointwise certificates for different settings. """
import torch
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Batch vs Pointwise Certification')
parser.add_argument('--evaluations',  type=str, help='name of evaluations directory')
parser.add_argument('--method',  type=str, help='method to compare: dpa, dpa_roe, fa, fa_roe')

parser.add_argument('--batch_size', type=int, default=100, help='Test batch size for certification')
parser.add_argument('--test_size', type=int, default=10000, help='Total test size for certification')
parser.add_argument('--k_poisons', type=int, nargs='+', help='Number of points the adversary can poison')

args = parser.parse_args()

if not os.path.exists('results'):
    os.makedirs('results')

""" DPA*+ROE Comparison """
batch_certs = []
batch_size = args.batch_size

k_poisons = args.k_poisons

for k_poison in k_poisons:
    batch_certs = []
    for i in range(0, args.test_size, batch_size):
        print(f"Loading batch {i} to {i+batch_size}")
        batch_cert = torch.load(f'batch_certs/{str(args.method)}_{str(args.evaluations)}/cert_accs_N={k_poison}_batch_{i}_{i+batch_size}.pth')
        print(batch_cert)
        print(f"k_poison: {k_poison}")
        cert_acc, opt_gap = batch_cert
        print(f"Certified accuracy: {cert_acc}")
        print(f"Optimality gap: {opt_gap}")
        batch_certs.append(cert_acc)

    batch_cert_acc = np.mean(batch_certs)
    batch_cert_max = np.amax(batch_certs)
    batch_cert_min = np.amin(batch_certs)
    batch_cert_std = np.std(batch_certs)
    print(f"Batch certs: {batch_cert_acc}")

    torch.save(
        {
            'batch_cert_acc': batch_cert_acc,
            'batch_cert_max': batch_cert_max,
            'batch_cert_min': batch_cert_min,
            'batch_cert_std': batch_cert_std,
        },
        f'results/r_{str(args.method)}_{str(args.evaluations)}_N={k_poison}.pth'
    )
    
    