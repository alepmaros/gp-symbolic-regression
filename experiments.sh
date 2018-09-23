#!/bin/bash

TIMESTAMP=$(date +%s)
SEED=8141516

# ### Synth 1
# echo "Synth 1"

# # Change in Population on Synth 1
# echo "Running with 50 Individuals"
# f1=$(python3 run.py -d synth1 -g 500 -k 5 -p 50 -r 30 -cp 0.85 -mp 0.14 -rp 0.01  --random-seed $SEED --timestamp $TIMESTAMP --test pop_test | tail -1)
# echo "Running with 100 Individuals"
# f2=$(python3 run.py -d synth1 -g 500 -k 5 -p 100 -r 30 -cp 0.85 -mp 0.14 -rp 0.01 --random-seed $SEED --timestamp $TIMESTAMP --test pop_test | tail -1)
# echo "Running with 200 Individuals"
# f3=$(python3 run.py -d synth1 -g 500 -k 5 -p 200 -r 30 -cp 0.85 -mp 0.14 -rp 0.01 --random-seed $SEED --timestamp $TIMESTAMP --test pop_test | tail -1)
# echo "Running with 500 Individuals"
# f4=$(python3 run.py -d synth1 -g 500 -k 5 -p 500 -r 30 -cp 0.85 -mp 0.14 -rp 0.01 --random-seed $SEED --timestamp $TIMESTAMP --test pop_test | tail -1)

# echo "Plotting Population Test"
# python3 plots/plot_population.py --scores $f1 $f2 $f3 $f4 --dataset synth1

# # Number of Generations Synth 1
# echo "Running 50 Generations on Synth 1"
# f1=$(python3 run.py -d synth1 -g 50 -k 5 -p 100 -r 30 -cp 0.85 -mp 0.14 -rp 0.01 --random-seed $SEED --timestamp $TIMESTAMP --test test_generations | tail -1)
# echo "Running 100 Generations on Synth 1"
# f2=$(python3 run.py -d synth1 -g 100 -k 5 -p 100 -r 30 -cp 0.85 -mp 0.14 -rp 0.01 --random-seed $SEED --timestamp $TIMESTAMP --test test_generations | tail -1)
# echo "Running 200 Generations on Synth 1"
# f3=$(python3 run.py -d synth1 -g 200 -k 5 -p 100 -r 30 -cp 0.85 -mp 0.14 -rp 0.01 --random-seed $SEED --timestamp $TIMESTAMP --test test_generations | tail -1)
# echo "Running 500 Generations on Synth 1"
# f4=$(python3 run.py -d synth1 -g 500 -k 5 -p 100 -r 30 -cp 0.85 -mp 0.14 -rp 0.01 --random-seed $SEED --timestamp $TIMESTAMP --test test_generations | tail -1)

# python3 plots/plot_test_generations.py --scores $f1 $f2 $f3 $f4  --dataset synth1

# # Operator Probabilities on Synth 1
# echo "Running operator Probability 1"
# f1=$(python3 run.py -d synth1 -g 200 -k 5 -p 100 -r 30 -cp 0.9 -mp 0.09 -rp 0.01 --random-seed $SEED --timestamp $TIMESTAMP --test operator_prob | tail -1)
# echo "Running operator Probability 2"
# f2=$(python3 run.py -d synth1 -g 200 -k 5 -p 100 -r 30 -cp 0.7 -mp 0.29 -rp 0.01 --random-seed $SEED --timestamp $TIMESTAMP --test operator_prob | tail -1)
# echo "Running operator Probability 3"
# f3=$(python3 run.py -d synth1 -g 200 -k 5 -p 100 -r 30 -cp 0.5 -mp 0.49 -rp 0.01 --random-seed $SEED --timestamp $TIMESTAMP --test operator_prob | tail -1)

# python3 plots/plot_operators_prob.py --scores $f1 $f2 $f3 --dataset synth1

# # Tournament Size on Synth 1
# echo "Running Tournament Size 2"
# f1=$(python3 run.py -d synth1 -g 200 -k 2 -p 100 -r 30 -cp 0.7 -mp 0.29 -rp 0.01 --random-seed $SEED --timestamp $TIMESTAMP --test tournament_size | tail -1)
# echo "Running Tournament Size 5"
# f2=$(python3 run.py -d synth1 -g 200 -k 5 -p 100 -r 30 -cp 0.7 -mp 0.29 -rp 0.01 --random-seed $SEED --timestamp $TIMESTAMP --test tournament_size | tail -1)
# echo "Running Tournament Size 7"
# f3=$(python3 run.py -d synth1 -g 200 -k 7 -p 100 -r 30 -cp 0.7 -mp 0.29 -rp 0.01 --random-seed $SEED --timestamp $TIMESTAMP --test tournament_size | tail -1)

# python3 plots/plot_tournament_size.py --scores $f1 $f2 $f3 --dataset synth1

# # Elists on Synth 1
# echo "Running Without Elitist Operators"
# f1=$(python3 run.py -d synth1 -g 200 -k 2 -p 100 -r 30 -cp 0.7 -mp 0.29 -rp 0.01 --random-seed $SEED --timestamp $TIMESTAMP --test elitist_operators | tail -1)
# echo "Running With Elistist Operators"
# f2=$(python3 run.py -d synth1 -g 200 -k 2 -p 100 -r 30 -cp 0.7 -mp 0.29 -rp 0.01 --elitist-operators --random-seed $SEED --timestamp $TIMESTAMP --test elitist_operators | tail -1)

# python3 plots/plot_elitist_operators.py --scores $f1 $f2 --dataset synth1

# # Best, Average Synth 1
# echo "Best Average on Synth 1"
# f1=$(python3 run.py -d synth1 -g 200 -k 2 -p 200 -r 30 -cp 0.5 -mp 0.49 -rp 0.01 --random-seed $SEED --timestamp $TIMESTAMP --test best_avg | tail -1)

# python3 plots/plot_avg_best.py --scores $f1 --dataset synth1

# # Synth 2
# echo "Synth 2"
# # Change in Population on Synth 2
# echo "Running with 50 Individuals"
# f1=$(python3 run.py -d synth2 -g 500 -k 5 -p 50 -r 30 -cp 0.85 -mp 0.10 -rp 0.05 --random-seed $SEED --timestamp $TIMESTAMP --test pop_test | tail -1)
# echo "Running with 100 Individuals"
# f2=$(python3 run.py -d synth2 -g 500 -k 5 -p 100 -r 30 -cp 0.85 -mp 0.10 -rp 0.05 --random-seed $SEED --timestamp $TIMESTAMP --test pop_test | tail -1)
# echo "Running with 200 Individuals"
# f3=$(python3 run.py -d synth2 -g 500 -k 5 -p 200 -r 30 -cp 0.85 -mp 0.10 -rp 0.05 --allow-sin --random-seed $SEED --timestamp $TIMESTAMP --test pop_test | tail -1)
# echo "Running with 500 Individuals"
# f4=$(python3 run.py -d synth2 -g 500 -k 5 -p 500 -r 30 -cp 0.85 -mp 0.10 -rp 0.05 --allow-sin --random-seed $SEED --timestamp $TIMESTAMP --test pop_test | tail -1)

# echo "Plotting Population Test"
# python3 plots/plot_population.py --scores $f1 $f2 $f3 $f4 --dataset synth2

# Allow Sin
echo "Running with not allow sin"
f1=$(python3 run.py -d synth2 -g 500 -k 2 -p 200 -r 30 -cp 0.5 -mp 0.49 -rp 0.01 --random-seed $SEED --timestamp $TIMESTAMP --test allow_sin | tail -1)
echo "Running with allow sin"
f2=$(python3 run.py -d synth2 -g 500 -k 2 -p 200 -r 30 -cp 0.5 -mp 0.49 -rp 0.01 --allow-sin --random-seed $SEED --timestamp $TIMESTAMP --test allow_sin | tail -1)

python3 plots/plot_sin.py --scores $f1 $f2 --dataset synth2

# Best, Average Synth 2
echo "Best Average on Synth 2"
f1=$(python3 run.py -d synth1 -g 500 -k 2 -p 200 -r 30 -cp 0.5 -mp 0.49 -rp 0.01 --allow-sin --random-seed $SEED --timestamp $TIMESTAMP --test best_avg | tail -1)

python3 plots/plot_avg_best.py --scores $f1 --dataset synth2

# Concrete
echo "Concrete"
echo "Running with not allow sin"
f1=$(python3 run.py -d concrete -g 500 -k 2 -p 200 -r 30 -cp 0.9 -mp 0.09 -rp 0.01 --random-seed $SEED --timestamp $TIMESTAMP --test allow_sin | tail -1)
echo "Running with allow sin"
f2=$(python3 run.py -d concrete -g 500 -k 2 -p 200 -r 30 -cp 0.9 -mp 0.09 -rp 0.01 --allow-sin --random-seed $SEED --timestamp $TIMESTAMP --test allow_sin | tail -1)

python3 plots/plot_sin.py --scores $f1 $f2 --dataset concrete

echo "Best Average on Synth 2"
f1=$(python3 run.py -d concrete -g 500 -k 2 -p 200 -r 30 -cp 0.9 -mp 0.09 -rp 0.01 --allow-sin --random-seed $SEED --timestamp $TIMESTAMP --test best_avg | tail -1)

python3 plots/plot_avg_best.py --scores $f1 --dataset concrete