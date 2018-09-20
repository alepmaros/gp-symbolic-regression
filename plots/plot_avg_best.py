import json,csv,time,os,sys,argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(description='Plot avg and best with std for run')
parser.add_argument('--scores', nargs='+', required='True', default='Scores path')
parser.add_argument('--dataset', type=str, required=True)

args = parser.parse_args()

if (len(args.scores) > 1):
    exit('Too many scores passed')

plt.style.use('ggplot')
f = plt.figure()
f.set_figheight(5)
f.set_figwidth(8)

for i, scores_path in enumerate(args.scores):
    with open(scores_path, 'r') as fhandle:
        score = (json.loads(fhandle.read()))
        
        label = 'Indivíduo Médio'

        scores_train_avg = [ x['Train']['Average'] for x in score['scores'] ]
        scores_train_best = [ x['Train']['Best'] for x in score['scores'] ]

        plt.plot(np.arange(0,len(scores_train_avg[0])), 
            np.mean(scores_train_avg, axis=0), 'b-',
            label=label)

        plt.fill_between(np.arange(0,len(scores_train_avg[0])),
                np.mean(scores_train_avg, axis=0) - np.std(scores_train_avg, axis=0),
                np.mean(scores_train_avg, axis=0) + np.std(scores_train_avg, axis=0),
                alpha=0.3, color='b')

        label = 'Melhor Indivíduo'

        plt.plot(np.arange(0,len(scores_train_best[0])), 
            np.mean(scores_train_best, axis=0), 'g-',
            label=label)

        plt.fill_between(np.arange(0,len(scores_train_best[0])),
                np.mean(scores_train_best, axis=0) - np.std(scores_train_best, axis=0),
                np.mean(scores_train_best, axis=0) + np.std(scores_train_best, axis=0),
                alpha=0.3, color='g')

        plt.title('{} - Fitness para dados de treino'.format(args.dataset))
        plt.xlabel('Geração')
        plt.ylabel('Fitness')
        plt.legend()


save_path, _ = os.path.split(args.scores[0])
save_path = os.path.join(save_path, '{}_best_avg.pdf'.format(args.dataset))
plt.tight_layout()
plt.savefig(save_path, format='pdf')