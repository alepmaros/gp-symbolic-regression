import os, argparse, sys, random

import numpy as np

from src.utils import read_dataset
from src.gp    import GeneticProgramming

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Genetic Programming to solve Symbolic Regression')
    parser.add_argument('--dataset', '-d', type=str, required=True,
                        choices=['synth1', 'synth2', 'concrete'],
                        help='Dataset to be used')
    parser.add_argument('--runs', '-r', type=int, default=10,
                        help='Number of runs to calculate mean and std')
    parser.add_argument('--generations', '-g', type=int, default=50,
                        help='Number of generations')
    parser.add_argument('--max-tree-depth', type=int, default=7,
                        help='The maximum depth of the function tree')
    parser.add_argument('--crossover-probability', '-c', type=float, default=0.9,
                        help='Crossover probability (mutation will be 1-p)')
    parser.add_argument('--individuals', '-i', type=int, default=50,
                        help='The number of individuals per generation')
    parser.add_argument('--random-seed', type=int, default=random.randint(0,1000000),
                        help='The seed for the random number generator')

    args = parser.parse_args()
    print(args)

    synth1_train = 'datasets/synth1/synth1-train.csv'
    synth1_test  = 'datasets/synth1/synth1-test.csv'
    synth2_train = 'datasets/synth2/synth2-train.csv'
    synth2_test  = 'datasets/synth2/synth2-test.csv'
    concrete_train = 'datasets/concrete/concrete-train.csv'
    concrete_test  = 'datasets/concrete/concrete-test.csv'

    train = []
    test  = []
    if (args.dataset == 'synth1'):
        train, test = read_dataset(synth1_train, synth1_test)
    elif (args.dataset == 'synth2'):
        train, test = read_dataset(synth2_train, synth2_test)
    elif (args.dataset == 'concrete'):
        train, test = read_dataset(concrete_train, concrete_test)
    else:
        exit('Invalid Dataset')

    rgenerator = np.random.RandomState(seed=args.random_seed)

    for i in range(0, args.runs):
        gp = GeneticProgramming(train, test,
                                args.individuals,
                                args.generations,
                                args.crossover_probability,
                                args.max_tree_depth,
                                rgenerator)
        scores = gp.run()