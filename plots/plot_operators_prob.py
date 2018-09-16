import json,csv,time,os,sys,argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Plot for different operators probabilities')
parser.add_argument('--scores', nargs='+', required='True', default='Scores path')
args = parser.parse_args()

if (len(args.scores) > 4):
    exit('Too many scores passed')

available_lines  = ['b-', 'r-', 'g-', 'k-', 'c-']
available_colors = ['b', 'r', 'g', 'k', 'c']

plt.style.use('ggplot')
plt.title('Fitness média com variação de probabilidade de operadores')

for i, scores_path in enumerate(args.scores):
    with open(scores_path, 'r') as fhandle:
        score = (json.loads(fhandle.read()))
        
        crossover_proability = score['Parameters']['Crossover Probability']
        mutation_probability = score['Parameters']['Mutation Probability']
        label = '{}% Crossover - {}% Mutation'.format(crossover_proability, mutation_probability)

        scores_train_avg = [ x['Train']['Average'] for x in score['scores'] ]

        plt.plot(np.arange(0,len(scores_train_avg[0])), 
            np.mean(scores_train_avg, axis=0), available_lines[i],
            label=label)

        plt.fill_between(np.arange(0,len(scores_train_avg[0])),
                np.mean(scores_train_avg, axis=0) - np.std(scores_train_avg, axis=0),
                np.mean(scores_train_avg, axis=0) + np.std(scores_train_avg, axis=0),
                alpha=0.3, color=available_colors[i])

plt.legend()
plt.show()

# scores_train_avg = [ x['Train']['Average'] for x in total_scores ]
# scores_train_best = [ x['Train']['Best'] for x in total_scores ]
# # scores_train_worst = [ x['Train']['Worst'] for x in total_scores ]
# plt.style.use('ggplot')
# plt.plot(np.arange(0,args.generations), 
#             np.mean(scores_train_avg, axis=0), 'b-')
# plt.plot(np.arange(0,args.generations), 
#             np.mean(scores_train_best, axis=0), 'g-')
# # plt.plot(np.arange(0,args.generations), 
# #             np.mean(scores_train_worst, axis=0), 'r-')
# plt.fill_between(np.arange(0,args.generations),
#                 np.mean(scores_train_avg, axis=0) - np.std(scores_train_avg, axis=0),
#                 np.mean(scores_train_avg, axis=0) + np.std(scores_train_avg, axis=0),
#                 alpha=0.4, color='b')
# plt.fill_between(np.arange(0,args.generations),
#                 np.mean(scores_train_best, axis=0) - np.std(scores_train_best, axis=0),
#                 np.mean(scores_train_best, axis=0) + np.std(scores_train_best, axis=0),
#                 alpha=0.4, color='g')
# # plt.fill_between(np.arange(0,args.generations),
# #                 np.mean(scores_train_worst, axis=0) - np.std(scores_train_worst, axis=0),
# #                 np.mean(scores_train_worst, axis=0) + np.std(scores_train_worst, axis=0),
# #                 alpha=0.4, color='r')
# print(scores_train_avg)
# plt.show()
