import matplotlib.pyplot as plt
import pickle
import os
import time

data_path = 'plots_plakat/plot_data/rl_policies_statistics.pkl'

with open(data_path, 'rb') as f:
    data = pickle.load(f)
    f.close()

x = range(1, 11)
plt.figure(figsize=(10, 6))

for key, value in data.items():
    print(f'{key}: {value[-1,0]}')
    mean = value[:,0]
    std = value[:,1]

    upper = mean + std
    lower = mean - std
    p = plt.plot(x, mean, label=key)
    c = p[-1].get_color()
    plt.fill_between(x, upper, lower, color=c, alpha=0.3)


plt.title('Example Plot of Different Datasets')
plt.xlabel('Steps')
plt.ylabel('Accuracy')

plt.legend()
plt.show()