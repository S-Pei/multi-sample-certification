#!/bin/bash
# redirect stdout and stderr to a file


# Run script to run batch certificartion
source /vol/bitbucket/spo21/myenv/bin/activate

eval=$1
output=$2
k=$3
d=$4
num_classes=$5
k_poison=$6

batch_size=$7
num_batches=$((5000/batch_size))


exec > "${output}.log" 2>&1

# Compute FA, FA+ROE certificates
# echo "Computing FA pointwise certificates"
# python3 pointwise-evaluation/fa_certify.py --evaluations=$eval --k=$k --d=$3 --num_classes=$num_classes
# echo "Computing FA+ROE pointwise certificates"
# python3 pointwise-evaluation/fa_roe_certify.py --evaluations=$eval --k=$k --d=$3 --num_classes=$num_classes

# Compute FA batch certificates in parallel
echo "Computing FA batch certificates"
echo "Starting script with batch size $batch_size and $num_batches batches from 0"
python3 batch_evaluation/batch_fa_certify.py --evaluations=$eval --k=$k --d=$d --num_classes=$num_classes --batch_size=$batch_size --from_idx=0 --num_batches=$num_batches --k_poisons $k_poison > "${output}_fa_1.log" 2>&1 &
echo "Starting script with batch size $batch_size and $num_batches batches from 5000"
python3 batch_evaluation/batch_fa_certify.py --evaluations=$eval --k=$k --d=$d --num_classes=$num_classes --batch_size=$batch_size --from_idx=5000 --num_batches=$num_batches --k_poisons $k_poison > "${output}_fa_2.log" 2>&1 &
wait

# Compute FA+ROE batch certificates in parallel
echo "Computing FA+ROE batch certificates"
echo "Starting script with batch size $batch_size and $num_batches batches from 0"
python3 batch_evaluation/batch_fa_roe_certify.py --evaluations=$eval --k=$k --d=$d --num_classes=$num_classes --batch_size=$batch_size --from_idx=0 --num_batches=$num_batches --k_poisons $k_poison > "${output}_fa_roe_1.log" 2>&1 &
echo "Starting script with batch size $batch_size and $num_batches batches from 5000"
python3 batch_evaluation/batch_fa_roe_certify.py --evaluations=$eval --k=$k --d=$d --num_classes=$num_classes --batch_size=$batch_size --from_idx=5000 --num_batches=$num_batches --k_poisons $k_poison > "${output}_fa_roe_2.log" 2>&1 &
wait

echo "All certifications done, computing comparisons"

python exp_scripts/comparison.py --evaluations=$eval --method=fa --batch_size=$batch_size --test_size=10000 --k_poisons $k_poison &
python exp_scripts/comparison.py --evaluations=$eval --method=fa_roe --batch_size=$batch_size --test_size=10000 --k_poisons $k_poison &
wait

# Print results
python exp_scripts/load_print.py --evaluations=$eval --method=fa --k_poisons $k_poison --k=$k --d=$d
python exp_scripts/load_print.py --evaluations=$eval --method=fa_roe --k_poisons $k_poison --k=$k --d=$d