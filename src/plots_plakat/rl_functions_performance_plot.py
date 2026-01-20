import matplotlib.pyplot as plt
import pickle
import os
import time

from color_shemes import main_colorscheme

x_steps = False

if x_steps:
    data_path = 'outputs/plots_plakat/plot_data/rl_policies_statistics.pkl'
else:
    data_path = 'outputs/plots_plakat/plot_data/rl_policies_statistics_gp.pkl'


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

i = 0
for key, value in data.items():
    if key not in data:
        continue

    mean = value[:,0]
    std = value[:,1]
    x = range(1, len(mean)+1)
    upper = mean + std
    lower = mean - std
    c = main_colorscheme[i]
    p = plt.plot(x, mean, label=key, color=c)
    #plt.fill_between(x, upper, lower, color=c, alpha=0.3)
    i += 1
    print(f'{key}: {value[-2,0]}')




plt.title('Performance of Different Models')
if x_steps:
    plt.xlabel('Steps')
else:
    plt.xlabel('Grasping Points')

plt.ylim((0, 100))
plt.xlim((1, 10))
plt.xlabel('Grasping points')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()

if x_steps:
    plt.savefig('outputs/plots_plakat/temp/performance_plot_steps.png', dpi=300)
else:
    plt.savefig('outputs/plots_plakat/temp/performance_plot_gps.png', dpi=300)

print("show")


plt.show()