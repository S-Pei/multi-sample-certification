# Scripts

We generally use the same code of work done by this [paper](https://arxiv.org/abs/2302.02300) <font size="2">(Run-Off Election: Improved Provable Defense against Data Poisoning Attacks)</font> in this [repo](https://github.com/k1rezaei/Run-Off-Election/tree/main?tab=readme-ov-file) to train base learners and compute pointwise certificates in our experiments.

### Assigning samples to training sets of different base learners
```
cd train
python3 FiniteAggregation_data_norm_hash.py --dataset=cifar --k=50 --d=16
```
Here `--dataset` can be `mnist`, `cifar` and `gtsrb`, which are benchmarks evaluated in our paper; `--k` and `--d` corresponds to the hyper-parameters for our FA/FA+ROE. `d=1` corresponds to DPA/DPA+ROE.

### Training the base learners
```
cd train
python3 FiniteAggregation_train_cifar_nin_baseline.py --k=50 --d=16 --start=0 --range=800
```
Here `--k` and `--d` are the same as above, and a total of $k\cdot d$ base learners will be trained independently. `--start` and `--range` specify which base learners are trained with this script.
For instance, when one uses `--k=50` and `--d=16`, one can use `--start=0` and `--range=800` to train all base learners sequentially, or one can use two separate runs with repsectively `--start=0` and `--start=400` (both with `--range=400`) to train in parallel the first 400 and the last 400 base learners.
To train on MNIST and GTSRB, run `FiniteAggregation_train_mnist_nin_baseline.py` and `FiniteAggregation_train_gtsrb_nin_baseline.py` respectively.


### Collecting predictions of base learners on test sets
```
python3 prediction/FiniteAggregation_evaluate_cifar_nin_baseline.py --models=cifar_nin_baseline_FiniteAggregation_k50_d16
```
For MNIST and GTSRB, run `FiniteAggregation_evaluate_mnist_nin_baseline.py` and `FiniteAggregation_evaluate_gtsrb_nin_baseline.py` instead.

### Computing the point-wise certified radius using the collected predictions
These three lines of codes, find the certified radius based on methods:
+ DPA+ROE
+ FA+ROE
+ FA.
```
python3 pointwise-evaluation/dpa_roe_certify.py --evaluations=cifar_nin_baseline_FiniteAggregation_k50_d1 --num_classes=10
python3 pointwise-evaluation/fa_roe_certify.py --evaluations=cifar_nin_baseline_FiniteAggregation_k50_d16 --k=50 --d=16 --num_classes=10
python3 pointwise-evaluation/fa_certify.py --evaluations=cifar_nin_baseline_FiniteAggregation_k50_d16 --k=50 --d=16 --num_classes=10
```

### Computing the multi-sample certified radius using the collected predictions
```
python3 batch_evaluation/batch_dpa_certify.py --evaluations=cifar_nin_baseline_FiniteAggregation_k50_d1 --num_classes=10 --batch_size=100 --from_idx=0 --num_batches=10
```
Here `--num_classes` is the size of the label set on the evalauted dataset (i.e. `--num_classes=10` for MNIST and CIFAR-10 and `--num_classes=43` for GTSRB) and
setting `--batch_size=1` corresponds to computing point-wise certificates. This allows certification to be done in batches over the whole dataset. For example, if one would like to compute the certification over the whole test dataset (of size 10000) in 10 batches, one can do 2 separate runs in parallel by setting: `--num_batches=10`, `--batch_size=1000` for both runs, and with `--from_idx=0` and `--from_idx=5000` respectively.

To find the certified batch accuracy using other methods:
+ DPA+ROE
+ FA+ROE
+ FA
```
python3 batch_evaluation/batch_dpa_roe_certify.py --evaluations=cifar_nin_baseline_FiniteAggregation_k50_d1 --num_classes=10 --batch_size=100 --from_idx=0 --num_batches=10
python3 batch_evaluation/batch_fa_roe_certify.py --evaluations=cifar_nin_baseline_FiniteAggregation_k50_d16 --k=50 --d=16 --num_classes=10 --batch_size=100 --from_idx=0 --num_batches=10
python3 batch_evaluation/batch_fa_certify.py --evaluations=cifar_nin_baseline_FiniteAggregation_k50_d16 --k=50 --d=16 --num_classes=10 --batch_size=100 --from_idx=0 --num_batches=10
```

