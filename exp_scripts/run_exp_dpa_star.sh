#!/bin/bash
# redirect stdout and stderr to a file


# Run script to run batch certificartion

eval=$1
output=$2
k=$3
d=$4
num_classes=$5
k_poison=$6

batch_size=$7
dataset=$8

test_size=10000
# split the workload into 2 processes
half=$((test_size / 2))
part1=$(( (half / batch_size) * batch_size ))
part2=$(( test_size - part1 ))
num_batches_1=$((part1/batch_size))
num_batches_2=$((part2/batch_size))
exec > "${output}.log" 2>&1

# Compute ROE point-wise certificates
echo "Computing ROE pointwise certificates"
python3 pointwise-evaluation/dpa_roe_certify_boosted.py --evaluations=$eval --k=$k --d=$d --num_classes=$num_classes
wait

# Compute DPA+ROE batch certificates in parallel
echo "Computing DPA*+ROE batch certificates"
SECONDS=0
echo "Starting script with batch size $batch_size and $num_batches_1 batches from 0"
python3 batch_evaluation/batch_dpa_star_roe_certify.py --evaluations=$eval --k=$k --d=$d --num_classes=$num_classes --batch_size=$batch_size --from_idx=0 --num_batches=$num_batches_1 --k_poisons $k_poison --dataset=$dataset > "${output}_dpa_star_roe_1.log" 2>&1 &
echo "Starting script with batch size $batch_size and $num_batches_2 batches from $part1"
python3 batch_evaluation/batch_dpa_star_roe_certify.py --evaluations=$eval --k=$k --d=$d --num_classes=$num_classes --batch_size=$batch_size --from_idx=$part1 --num_batches=$num_batches_2 --k_poisons $k_poison --dataset=$dataset > "${output}_dpa_star_roe_2.log" 2>&1 &
wait

echo "Elapsed time: ${SECONDS} seconds"
max_num_batches=$(( num_batches_1 > num_batches_2 ? num_batches_1 : num_batches_2 ))
echo "average time per batch: $((SECONDS/max_num_batches)) seconds"

echo "All certifications done, computing comparisons"

python exp_scripts/comparison_dpa_star.py --evaluations=$eval --method=dpa_star_roe --batch_size=$batch_size --test_size=10000 --k_poisons $k_poison &