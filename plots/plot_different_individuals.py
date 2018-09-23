import json,csv,time,os,sys,argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Plot for different population counts')
parser.add_argument('--scores', nargs='+', required='True', default='Scores path')
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--individual', type=int, required=True)

args = parser.parse_args()

if (len(args.scores) > 4):
    exit('Too many scores passed')

available_lines  = ['b-', 'r-', 'g-', 'k-', 'c-']
available_colors = ['b', 'r', 'g', 'k', 'c']

plt.style.use('ggplot')
f = plt.figure()
f.set_figheight(5)
f.set_figwidth(8)

for i, scores_path in enumerate(args.scores):
    with open(scores_path, 'r') as fhandle:
        score = (json.loads(fhandle.read()))
        
        tournament_size = score['Parameters']['Tournament Size']
        label = r'$k$ = {}'.format(tournament_size)

        different_individuals = [ x['Train']['Different Individuals'] for x in score['scores'] ]

        plt.plot(np.arange(0,len(different_individuals[0])), 
            np.mean(different_individuals, axis=0), available_lines[i],
            label=label)

        plt.fill_between(np.arange(0,len(different_individuals[0])),
                np.mean(different_individuals, axis=0) - np.std(different_individuals, axis=0),
                np.mean(different_individuals, axis=0) + np.std(different_individuals, axis=0),
                alpha=0.3, color=available_colors[i])

        plt.title('{} Número de indivíduos distintos \nVariação do $k$ - {} Indivíduos'.format(args.dataset, args.individual))
        plt.xlabel('Geração')
        plt.ylabel('Número de indivíduos distintos')
        plt.legend()


save_path, _ = os.path.split(args.scores[0])
save_path = os.path.join(save_path, '{}_different_individuals_train.pdf'.format(args.dataset))
plt.tight_layout()
plt.savefig(save_path, format='pdf')
