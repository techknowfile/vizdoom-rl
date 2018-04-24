from matplotlib import pyplot as plt
import pandas as pd
#plt.style.use('fivethirtyeight')
plt.style.use('ggplot')
import json
import os, glob

"""
:summary: This plots the data file saved with option = 1. 
This entire would was written
"""

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)


def add_to_subplot(ax, model, label, label_var, c=0):
    with open(model) as f:
        data=json.load(f)
        rolmean = pd.rolling_mean(pd.Series(data['means'], data['learning_steps']), 12, 1)
    ax.plot(data['learning_steps'], data['means'], c=tableau20[c+1], alpha=0.7, linewidth=2.0)
    ax.plot(rolmean, c=tableau20[c], alpha=0.8, linewidth=2.0, label=label + ': ' + str(label_var))
    ax.legend(loc='best')


#--- CODE TO PLOT one plot
add_to_subplot(plt,'./data/model-gamedtc-fr3-kf3-long.pth', 'average_score', 2, 4)
plt.draw()
plt.show()
input('press [enter] to close.')
