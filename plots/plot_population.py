import json,csv,time,os,sys,argparse
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Plot for different population counts')
parser.add_argument('--scores', nargs='+', required='True', default='Scores path')
parser.add_argument('--dataset', type=str, required=True)

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
        
        individuals = score['Parameters']['Population']
        label = r'${} Indivíduos'.format(individuals)

        scores_train_avg = [ x['Train']['Average'] for x in score['scores'] ]

        plt.plot(np.arange(0,len(scores_train_avg[0])), 
            np.mean(scores_train_avg, axis=0), available_lines[i],
            label=label)

        # plt.fill_between(np.arange(0,len(scores_train_avg[0])),
        #         np.mean(scores_train_avg, axis=0) - np.std(scores_train_avg, axis=0),
        #         np.mean(scores_train_avg, axis=0) + np.std(scores_train_avg, axis=0),
        #         alpha=0.3, color=available_colors[i])

        plt.title('Fitness Média para dados de Treino')
        plt.xlabel('Geração')
        plt.ylabel('Fitness')
        plt.legend()


save_path, _ = os.path.split(args.scores[0])
save_path = os.path.join(save_path, '{}_operators_prob_train.pdf'.format(args.dataset))
plt.tight_layout()
plt.savefig(save_path, format='pdf')

# Best Individuals

plt.style.use('ggplot')
f, axs = plt.subplots(1, 2, sharex='col', sharey='row')
f.set_figheight(5)
f.set_figwidth(11)

ax0 = axs.reshape(-1)[0]
ax1 = axs.reshape(-1)[1]

# Best Individual Train

xtick_labels = []
list_scores_test = []
for i, scores_path in enumerate(args.scores):
    with open(scores_path, 'r') as fhandle:
        score = (json.loads(fhandle.read()))
        
        individuals = score['Parameters']['Population']
        label = r'{} Indivíduos'.format(individuals)
        
        scores_train_best = [ x['Train']['Best'][-1] for x in score['scores'] ]
        list_scores_test.append(np.ravel(scores_train_best[:-1]))
        xtick_labels.append(label)

ax0.boxplot(list_scores_test)
print(xtick_labels)
ax0.set_xticklabels(xtick_labels, rotation=15)
ax0.set_title('Fitness do melhor indivíduo na última geração\npara base de Treino')

# Best Individual Test
xtick_labels = []
list_scores_test = []
for i, scores_path in enumerate(args.scores):
    with open(scores_path, 'r') as fhandle:
        score = (json.loads(fhandle.read()))
        
        individuals = score['Parameters']['Population']
        label = r'{} individuals'.format(individuals)
        
        scores_test_best = [ x['Test']['Best'] for x in score['scores'] ]
        list_scores_test.append(np.ravel(scores_test_best))
        xtick_labels.append(label)

ax1.boxplot(list_scores_test)
print(xtick_labels)
ax1.set_xticklabels(xtick_labels, rotation=15)
ax1.set_title('Fitness do melhor indivíduo na última geração\npara base de Teste')


# save_path, _ = os.path.split(args.scores[0])
# save_path = os.path.join(save_path, '{}_operators_prob_best_test.pdf'.format(args.dataset))
# plt.tight_layout()
# plt.savefig(save_path, format='pdf')


save_path, _ = os.path.split(args.scores[0])
save_path = os.path.join(save_path, '{}_operators_prob_best.pdf'.format(args.dataset))
plt.tight_layout()
plt.savefig(save_path, format='pdf')