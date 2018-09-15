import os, argparse, sys, random

import matplotlib.pyplot as plt
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
    parser.add_argument('--crossover-probability', '-cp', type=float, default=0.90,
                        help='Crossover probability')
    parser.add_argument('--mutation-probability', '-mp', type=float, default=0.05,
                        help='Mutation Probability')
    parser.add_argument('--reproduction-probability', '-rp', type=float, default=0.05,
                        help='Reprodution Probability')
    parser.add_argument('--population', '-p', type=int, default=50,
                        help='The number of the population per generation')
    parser.add_argument('--tournament-size', '-k', type=int, default=10,
                        help='How many individuals will be selected in the tournament')
    parser.add_argument('--elitist-operators', '-e', type=bool, default=False,
                        help='If Elitist operators are enabled')
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

    total_scores = []
    for i in range(0, args.runs):
        print('Run', i)
        gp = GeneticProgramming(train, test,
                                args.population,
                                args.generations,
                                args.crossover_probability,
                                args.mutation_probability,
                                args.reproduction_probability,
                                args.max_tree_depth,
                                args.tournament_size,
                                rgenerator)
        scores = gp.run()
        total_scores.append(scores)

    scores_train_avg = [ x['Train']['Average'] for x in total_scores ]
    scores_train_best = [ x['Train']['Best'] for x in total_scores ]
    # scores_train_worst = [ x['Train']['Worst'] for x in total_scores ]
    plt.style.use('ggplot')
    plt.plot(np.arange(0,args.generations), 
                np.mean(scores_train_avg, axis=0), 'b-')
    plt.plot(np.arange(0,args.generations), 
                np.mean(scores_train_best, axis=0), 'g-')
    # plt.plot(np.arange(0,args.generations), 
    #             np.mean(scores_train_worst, axis=0), 'r-')
    plt.fill_between(np.arange(0,args.generations),
                    np.mean(scores_train_avg, axis=0) - np.std(scores_train_avg, axis=0),
                    np.mean(scores_train_avg, axis=0) + np.std(scores_train_avg, axis=0),
                    alpha=0.4, color='b')
    plt.fill_between(np.arange(0,args.generations),
                    np.mean(scores_train_best, axis=0) - np.std(scores_train_best, axis=0),
                    np.mean(scores_train_best, axis=0) + np.std(scores_train_best, axis=0),
                    alpha=0.4, color='g')
    # plt.fill_between(np.arange(0,args.generations),
    #                 np.mean(scores_train_worst, axis=0) - np.std(scores_train_worst, axis=0),
    #                 np.mean(scores_train_worst, axis=0) + np.std(scores_train_worst, axis=0),
    #                 alpha=0.4, color='r')
    print(scores_train_avg)
    plt.show()

    