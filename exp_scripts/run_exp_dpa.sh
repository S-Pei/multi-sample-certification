#!/bin/bash
# redirect stdout and stderr to a file


# Run script to run batch certificartion
source /vol/bitbucket/spo21/myenv/bin/activate

eval=$1
output=$2
k=$3
num_classes=$4
k_poison=$5

batch_size=$6

test_size=10000
# split the workload into 2 processes
half=$((test_size / 2))
part1=$(( (half / batch_size) * batch_size ))
part2=$(( test_size - part1 ))
num_batches_1=$((part1/batch_size))
num_batches_2=$((part2/batch_size))


exec > "${output}.log" 2>&1

# Compute DPA, DPA+ROE pointwise certificates
# echo "Computing DPA, DPA+ROE pointwise certificates"
# python3 pointwise-evaluation/dpa_roe_certify.py --evaluations=$eval --num_classes=$num_classes

# Compute DPA batch certificates in parallel
echo "Computing DPA batch certificates"
echo "Starting script with batch size $batch_size and $num_batches_1 batches from 0"
python3 batch_evaluation/batch_dpa_certify.py --evaluations=$eval --num_classes=$num_classes --batch_size=$batch_size --from_idx=0 --num_batches=$num_batches_1 --k_poisons $k_poison > "${output}_dpa_1.log" 2>&1 &
echo "Starting script with batch size $batch_size and $num_batches_2 batches from $part1"
python3 batch_evaluation/batch_dpa_certify.py --evaluations=$eval --num_classes=$num_classes --batch_size=$batch_size --from_idx=$part1 --num_batches=$num_batches_2 --k_poisons $k_poison > "${output}_dpa_2.log" 2>&1 &
wait

# Compute DPA+ROE batch certificates in parallel
echo "Computing DPA+ROE batch certificates"
echo "Starting script with batch size $batch_size and $num_batches_1 batches from 0"
python3 batch_evaluation/batch_dpa_roe_certify.py --evaluations=$eval --num_classes=$num_classes --batch_size=$batch_size --from_idx=0 --num_batches=$num_batches_1 --k_poisons $k_poison > "${output}_dpa_roe_1.log" 2>&1 &
echo "Starting script with batch size $batch_size and $num_batches_2 batches from $part1"
python3 batch_evaluation/batch_dpa_roe_certify.py --evaluations=$eval --num_classes=$num_classes --batch_size=$batch_size --from_idx=$part1 --num_batches=$num_batches_2 --k_poisons $k_poison > "${output}_dpa_roe_2.log" 2>&1 &
wait

echo "All certifications done, computing comparisons"

python exp_scripts/comparison.py --evaluations=$eval --method=dpa --batch_size=$batch_size --test_size=$test_size --k_poisons $k_poison &
python exp_scripts/comparison.py --evaluations=$eval --method=dpa_roe --batch_size=$batch_size --test_size=$test_size --k_poisons $k_poison &
wait

# Print results
python exp_scripts/load_print.py --evaluations=$eval --method=dpa --k_poisons $k_poison --k=$k --d=1
python exp_scripts/load_print.py --evaluations=$eval --method=dpa_roe --k_poisons $k_poison --k=$k --d=1