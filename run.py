import os, argparse, sys, random, time, json

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
    parser.add_argument('--max-tree-depth', type=int, default=6,
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
    parser.add_argument('--timestamp', type=str, default=str(time.time()).split('.')[0],
                        help='Timestamp of when the experiment is being run, to aggregate same experiments into one folder.')

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

    all_runs = {
        'scores': [],
        'Parameters': {
            'Dataset': args.dataset,
            'Population': args.population,
            'Generations': args.generations,
            'Crossover Probability': args.crossover_probability,
            'Mutation Probability': args.mutation_probability,
            'Reproduction Probability': args.reproduction_probability,
            'Max Tree Depth': args.max_tree_depth,
            'Tournament Size': args.tournament_size,
            'Random Seed': args.random_seed
        }
    } 
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
        all_runs['scores'].append(scores)

    ## Save Runs
    save_directory = 'experiments/{}'.format(args.timestamp)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    file_name = 'scores_{}_pop{}_gen{}_cross{}_mut{}_repro{}_mtd{}_k{}_seed{}.json'.format(
        args.dataset,
        args.population,
        args.generations,
        args.crossover_probability,
        args.mutation_probability,
        args.reproduction_probability,
        args.max_tree_depth,
        args.tournament_size,
        args.random_seed
    )

    with open(os.path.join(save_directory, file_name), 'w') as fhandle:
        fhandle.write(json.dumps(all_runs, indent=2))

    print(os.path.join(save_directory, file_name))
    