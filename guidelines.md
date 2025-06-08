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

To train DPA* learners, `--start` and `--range` are defined relative to `k`. Eg. running
```
python3 FiniteAggregation_train_cifar_nin_boosted.py --k=50 --d=32 --start=0 --range=25
python3 FiniteAggregation_train_cifar_nin_boosted.py --k=50 --d=32 --start=25 --range=25
```
will train 25*32 learners each in parallel.
To train DPA* on MNIST and GTSRB, run 
`FiniteAggregation_train_mnist_nin_boosted.py` and `FiniteAggregation_train_gtsrb_nin_boosted.py` respectively.

### Collecting predictions of base learners on test sets
```
python3 prediction/FiniteAggregation_evaluate_cifar_nin_baseline.py --models=cifar_nin_baseline_FiniteAggregation_k50_d16
```
For MNIST and GTSRB, run `FiniteAggregation_evaluate_mnist_nin_baseline.py` and `FiniteAggregation_evaluate_gtsrb_nin_baseline.py` instead.

### Computing the point-wise certified radius using the collected predictions
Note that, to compute the **multi-sample certified radius** for aggregations that involve ROE (DPA+ROE, DPA*+ROE, FA+ROE), the DPA/DPA*+ROE **point-wise** certificates need to be computed before running the batch certification scripts (outlined in the next section). This is because the G values (refer to Section 4.1 of the paper for more details) are computed directly from the point-wise certificates. These commands also recover the point-wise certificates from previous works.\
For DPA+ROE, run:
```
python3 pointwise-evaluation/dpa_roe_certify.py --evaluations=cifar_nin_baseline_FiniteAggregation_k50_d1 --num_classes=10
```
For DPA*+ROE, run:
```
python3 pointwise-evaluation/dpa_roe_certify_boosted.py --evaluations=cifar_nin_baseline_FiniteAggregation_k50_d16_boosted --k=50 --d=16 --num_classes=10
```
For FA+ROE, run the DPA+ROE certificates as well with a different evaluations path:
```
python3 pointwise-evaluation/dpa_roe_certify.py --evaluations=cifar_nin_baseline_FiniteAggregation_k50_d16 --num_classes=10
```
Although not needed for batch certification, the respective FA/FA+ROE point-wise certificates from previous works can be computed similarly to find the point-wise certified radius based on methods:
+ FA+ROE
+ FA
```
python3 pointwise-evaluation/fa_roe_certify.py --evaluations=cifar_nin_baseline_FiniteAggregation_k50_d16 --k=50 --d=16 --num_classes=10
python3 pointwise-evaluation/fa_certify.py --evaluations=cifar_nin_baseline_FiniteAggregation_k50_d16 --k=50 --d=16 --num_classes=10
```

### Computing the multi-sample certified radius using the collected predictions
```
python3 batch_evaluation/batch_dpa_certify.py --evaluations=cifar_nin_baseline_FiniteAggregation_k50_d1 --num_classes=10 --dataset=cifar --batch_size=100 --from_idx=0 --num_batches=10 --k_poisons 3
```
Here `--num_classes` is the size of the label set on the evalauted dataset (i.e. `--num_classes=10` for MNIST and CIFAR-10 and `--num_classes=43` for GTSRB) and
setting `--batch_size=1` corresponds to computing point-wise certificates. This allows certification to be done in batches over the whole dataset. For example, if one would like to compute the certification over the whole test dataset (of size 10000) in 10 batches, one can do 2 separate runs in parallel by setting: `--num_batches=10`, `--batch_size=1000` for both runs, and with `--from_idx=0` and `--from_idx=5000` respectively.

To find the certified batch accuracy using other methods:
+ DPA+ROE
+ FA+ROE
+ FA
```
python3 batch_evaluation/batch_dpa_roe_certify.py --evaluations=cifar_nin_baseline_FiniteAggregation_k50_d1 --num_classes=10 --dataset=cifar --batch_size=100 --from_idx=0 --num_batches=10 --k_poisons 3
python3 batch_evaluation/batch_fa_roe_certify.py --evaluations=cifar_nin_baseline_FiniteAggregation_k50_d16 --k=50 --d=16 --num_classes=10 --dataset=cifar --batch_size=100 --from_idx=0 --num_batches=10 --k_poisons 3
python3 batch_evaluation/batch_fa_certify.py --evaluations=cifar_nin_baseline_FiniteAggregation_k50_d16 --k=50 --d=16 --num_classes=10 dataset=cifar --batch_size=100 --from_idx=0 --num_batches=10 --k_poisons 3
```

### Running experiments
To run experiments, there are 3 scripts:
+ To run DPA and DPA+ROE batch certification:
```
exp_scripts/run_exp_dpa.sh cifar_nin_baseline_FiniteAggregation_k50_d1 $eval_file_path $output_log_file $k $num_classes $k_poison $batch_size $dataset
```
For example:
```
exp_scripts/run_exp_dpa.sh cifar_nin_baseline_FiniteAggregation_k50_d1 logs/cifar_k=50_d=1/cifar_k=50_N=10_b=400 50 10 10 400 cifar
```
+ To run FA and FA+ROE batch certification:
```
exp_scripts/run_exp_fa.sh cifar_nin_baseline_FiniteAggregation_k50_d16 $eval_file_path $output_log_file $k $d $num_classes $k_poison $batch_size $dataset
```
For example:
```
exp_scripts/run_exp_fa.sh cifar_nin_baseline_FiniteAggregation_k50_d16 logs/cifar_k=50_d=16/cifar_k=50_d=16_N=5_b=100_1 50 16 10 5 100 cifar
```
+ To run DPA*+ROE batch certification:
```
exp_scripts/run_exp_dpa_star.sh cifar_nin_baseline_FiniteAggregation_k50_d16_boosted $eval_file_path $output_log_file $k $d $num_classes $k_poison $batch_size $dataset
```
For example:
```
exp_scripts/run_exp_dpa_star.sh cifar_nin_baseline_FiniteAggregation_k50_d16_boosted logs/cifar_k=50_d=16/cifar_k=50_d=16_N=5_b=100_1 50 16 10 5 100 cifar
```
Note that here, `$eval` is the same as the `--evaluations=` path used when computing point-wise/batch certificates as outlined in the previous sections. These scripts carries out 2 parallel runs of batch certificates over the whole testset. The results of the run is saved in the `results/` directory. The script also outputs a main log file, and separate log files for each run. Eg. `${output_path}_dpa_1.log`, `${output_path}_dpa_2.log`, `${output_path}_dpa_roe_1.log`,... Note that batch size should be divisible by the total test size (10000).

