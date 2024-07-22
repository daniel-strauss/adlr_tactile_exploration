import matplotlib.pyplot as plt
import pickle
import os
import time

from color_shemes import main_colorscheme

data_path = 'plots_plakat/plot_data/rl_policies_statistics_gp.pkl'

name_mask = {
    'random': 'Rand',
    'rl_models/rew500k9': 'Diff',
    'rl_models/obs500k9': 'Total',
    'rl_models/complex_after_free': 'TotalMiss',
    'rl_models/diff_after_free': 'preDiffMiss'
}


with open(data_path, 'rb') as f:
    data = pickle.load(f)
    f.close()

x = range(1, 11)
plt.figure(figsize=(8, 4))

counter = 0
for key in name_mask.keys():
    if key not in data:
        continue
    value = data[key]
    print(f'{key}: {value[-1,0]}')
    mean = value[:,0]
    std = value[:,1]

    upper = mean + std
    lower = mean - std
    c = main_colorscheme[counter]
    p = plt.plot(x, mean, label=name_mask[key], color=c)
    counter += 1
    #plt.fill_between(x, upper, lower, color=c, alpha=0.3)

plt.ylim((0, 100))
plt.xlim((1, 10))
plt.xlabel('Grasping points')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()

plt.savefig('plots_plakat/temp/performance_plot_gp.png', dpi=300)
plt.show()