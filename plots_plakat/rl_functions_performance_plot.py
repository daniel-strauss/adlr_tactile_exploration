import matplotlib.pyplot as plt
import pickle
import os
import time

save_path = 'plots_plakat/plot_data/rl_policies_statistics.pkl'

with open(save_path, 'rb') as f:
    data = pickle.load(f)
    f.close()



x = range(1, 11)
plt.figure(figsize=(10, 6))

for key, value in data.items():
    plt.plot(x, value, label=key)

plt.title('Example Plot of Different Datasets')
plt.xlabel('Steps')
plt.ylabel('Accuracy')

plt.legend()
plt.show()

time.sleep(30)