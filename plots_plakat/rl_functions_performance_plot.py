import matplotlib.pyplot as plt
import pickle
import os
import time

from plots_plakat.color_shemes import main_colorscheme

x_steps = False

if x_steps:
    data_path = 'plots_plakat/plot_data/rl_policies_statistics.pkl'
else:
    data_path = 'plots_plakat/plot_data/rl_policies_statistics_gp.pkl'

with open(data_path, 'rb') as f:
    data = pickle.load(f)
    f.close()

x = range(1, 11)
plt.figure(figsize=(10, 6))

for i, [key, value] in enumerate(data.items()):
    if not key.endswith('metric'):
        print(f'{key}: {value[-1,0]}')
        mean = value[:,0]
        std = value[:,1]

        upper = mean + std
        lower = mean - std
        c = main_colorscheme[i]
        p = plt.plot(x, mean, label=key, color=c)
        #plt.fill_between(x, upper, lower, color=c, alpha=0.3)


plt.title('Performance of Different Models')
if x_steps:
    plt.xlabel('Steps')
else:
    plt.xlabel('Grasping Points')

plt.ylabel('Accuracy')

plt.legend()
if x_steps:
    plt.savefig('plots_plakat/temp/performance_plot_steps.png')
else:
    plt.savefig('plots_plakat/temp/performance_plot_gps.png')


plt.show()