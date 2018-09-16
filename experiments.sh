#!/bin/bash

TIMESTAMP=$(date +%s)
SEED=8141516

### Synth 1
echo "Synth 1"
# Operator Probabilities on Synth 1
echo "Running operator Probability 1"
f1=$(python3 run.py --random-seed $SEED --timestamp $TIMESTAMP -d synth1 -g 300 -k 10 -p 100 -r 20 -cp 0.9 -mp 0.05 -rp 0.05 | tail -1)
echo "Running operator Probability 2"
f2=$(python3 run.py --random-seed $SEED --timestamp $TIMESTAMP -d synth1 -g 300 -k 10 -p 100 -r 20 -cp 0.7 -mp 0.25 -rp 0.05 | tail -1)
echo "Running operator Probability 3"
f3=$(python3 run.py --random-seed $SEED --timestamp $TIMESTAMP -d synth1 -g 300 -k 10 -p 100 -r 20 -cp 0.5 -mp 0.45 -rp 0.05 | tail -1)

python3 plots/plot_operators_prob.py --scores $f1 $f2 $f3
