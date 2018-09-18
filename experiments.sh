#!/bin/bash

TIMESTAMP=$(date +%s)
SEED=814151623

### Synth 1
echo "Synth 1"

# Change in Population on Synth 1
echo "Running with 50 Individuals"
f1=$(python3 run.py -d synth1 -g 300 -k 5 -p 50 -r 20 -cp 0.85 -mp 0.10 -rp 0.05  --random-seed $SEED --timestamp $TIMESTAMP --test pop_test | tail -1)
echo "Running with 100 Individuals"
f2=$(python3 run.py -d synth1 -g 300 -k 5 -p 100 -r 20 -cp 0.85 -mp 0.10 -rp 0.05 --random-seed $SEED --timestamp $TIMESTAMP --test pop_test | tail -1)
echo "Running with 500 Individuals"
f3=$(python3 run.py -d synth1 -g 300 -k 5 -p 500 -r 20 -cp 0.85 -mp 0.10 -rp 0.05 --random-seed $SEED --timestamp $TIMESTAMP --test pop_test | tail -1)

python3 plots/plot_population.py --scores $f1 $f2 $f3 --dataset synth1

# # Operator Probabilities on Synth 1
# echo "Running operator Probability 1"
# f1=$(python3 run.py -d synth1 -g 300 -k 30 -p 100 -r 30 -cp 0.9 -mp 0.05 -rp 0.05 --random-seed $SEED --timestamp $TIMESTAMP --test operator_prob | tail -1)
# echo "Running operator Probability 2"
# f2=$(python3 run.py -d synth1 -g 300 -k 30 -p 100 -r 30 -cp 0.7 -mp 0.25 -rp 0.05 --random-seed $SEED --timestamp $TIMESTAMP --test operator_prob | tail -1)
# echo "Running operator Probability 3"
# f3=$(python3 run.py -d synth1 -g 300 -k 30 -p 100 -r 30 -cp 0.5 -mp 0.45 -rp 0.05 --random-seed $SEED --timestamp $TIMESTAMP --test operator_prob | tail -1)

# python3 plots/plot_operators_prob.py --scores $f1 $f2 $f3 --dataset synth1
