import matplotlib.pyplot as plt
import pickle
import os
import time

from plots_plakat.color_shemes import main_colorscheme

data_path = 'plots_plakat/plot_data/rl_policies_statistics.pkl'

with open(data_path, 'rb') as f:
    data = pickle.load(f)
    f.close()

x = range(1, 11)
plt.figure(figsize=(10, 6))

for i, [key, value] in enumerate(data.items()):
    print(f'{key}: {value[-1,0]}')
    mean = value[:,0]
    std = value[:,1]

    upper = mean + std
    lower = mean - std
    c = main_colorscheme[i]
    p = plt.plot(x, mean, label=key, color=c)
    #plt.fill_between(x, upper, lower, color=c, alpha=0.3)


plt.title('Example Plot of Different Datasets')
plt.xlabel('Steps')
plt.ylabel('Accuracy')

plt.legend()

plt.savefig('plots_plakat/temp/performance_plot.png')
plt.show()