import matplotlib.pyplot as plt
import pandas as pd
import os

paths = [
    'outputs/plots_plakat/plot_data/obs500k/'
    'outputs/plots_plakat/plot_data/rew500k/'
]

for i in range(1, 11):
    

df = pd.read_csv