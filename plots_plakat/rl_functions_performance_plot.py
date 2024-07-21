import matplotlib.pyplot as plt
import pickle
import os
import time

from color_shemes import main_colorscheme

'''
x_steps = False

if x_steps:
    data_path = 'plots_plakat/plot_data/rl_policies_statistics.pkl'
else:
    data_path = 'plots_plakat/plot_data/rl_policies_statistics_gp.pkl'
'''
data_path = 'plots_plakat/plot_data/rl_policies_statistics_gp.pkl'

name_mask = {
    'rl_models/rew500k9': 'Diff',
    'rl_models/obs500k9': 'Abs',
    'random': 'Rand',
    'rl_models/allow_miss_free_ray': 'Miss',
    'rl_models/complex_after_free': 'AbsMiss',
    'rl_models/diff_after_free': 'DiffMiss'
}


with open(data_path, 'rb') as f:
    data = pickle.load(f)
    f.close()

x = range(1, 11)
plt.figure(figsize=(12, 6))

for i, [key, value] in enumerate(data.items()):
    '''
    if not key.endswith('metric'):
        print(f'{key}: {value[-1,0]}')
        mean = value[:,0]
        std = value[:,1]

        upper = mean + std
        lower = mean - std
        c = main_colorscheme[i]
        p = plt.plot(x, mean, label=key, color=c)
        #plt.fill_between(x, upper, lower, color=c, alpha=0.3)
    '''

    print(f'{key}: {value[-2,0]}')
    mean = value[:,0]
    std = value[:,1]

    upper = mean + std
    lower = mean - std
    c = main_colorscheme[i]
    p = plt.plot(x, mean, label=name_mask[key]), color=c)
    #plt.fill_between(x, upper, lower, color=c, alpha=0.3)

plt.title('Performance of Different Models')
if x_steps:
    plt.xlabel('Steps')
else:
    plt.xlabel('Grasping Points')


plt.ylim((0, 100))
plt.xlim((1, 10))
plt.title('Policies')
plt.xlabel('Steps')

plt.ylabel('Accuracy')

plt.legend()
if x_steps:
    plt.savefig('plots_plakat/temp/performance_plot_steps.png')
else:
    plt.savefig('plots_plakat/temp/performance_plot_gps.png')


plt.show()