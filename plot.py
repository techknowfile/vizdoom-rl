from matplotlib import pyplot as plt
import pandas as pd
#plt.style.use('fivethirtyeight')
plt.style.use('ggplot')
import json
import os, glob

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

#for i, result in enumerate(results):
#    print(result['means'])
#    plt.plot(result['learning_steps'], result['means'], label=result['experiment'])
plt.ion()
plt.show()
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

def add_to_subplot(ax, model, label, label_var, c=0):
    with open(model) as f:
        data=json.load(f)
        rolmean = pd.rolling_mean(pd.Series(data['means'], data['learning_steps']), 12, 1)
    ax.plot(data['learning_steps'], data['means'], c=tableau20[c+1], alpha=0.7, linewidth=2.0)
    ax.plot(rolmean, c=tableau20[c], alpha=0.8, linewidth=2.0, label=label + ': ' + str(label_var))
    ax.legend(loc='best')

add_to_subplot(ax1, './data/model-dtc-fr10-kf1-act3.pth', 'Actions', 3, 2)
add_to_subplot(ax1, './data/model-dtc-fr10-kf1.pth', 'Actions', 8, 18)

ax1.set_xlabel('Learning Steps', fontsize=12)
ax1.set_ylabel('Mean Score (100 episodes)', fontsize=12)
ax1.set_title('Frame Repeat: 10', fontsize=10)

add_to_subplot(ax2, './data/model-dtc-fr2-kf1-act3.pth', 'Actions', 3, 2)
add_to_subplot(ax2, './data/model-dtc-fr2-kf1.pth', 'Actions', 8, 18)
ax2.set_title('Frame Repeat: 2', fontsize=10)

plt.suptitle('Model Performance Over Time in Different Action Spaces', fontsize=13)
f.set_size_inches((11, 5))
plt.draw()
input('press [enter] to close.')
plt.close()

plt.figure(figsize=(5, 5))
add_to_subplot(plt, './data/model-dtc-fr2-kf1.pth', 'Frame Repeat', 2, 4)
add_to_subplot(plt, './data/model-dtc-fr5-kf1.pth', 'Frame Repeat', 5, 2)
add_to_subplot(plt, './data/model-dtc-fr10-kf1.pth', 'Frame Repeat', 10, 18)
add_to_subplot(plt, './data/model-dtc-fr20-kf1.pth', 'Frame Repeat', 20, 6)
plt.xlabel('Learning Steps', fontsize=12)
plt.ylabel('Mean Score (100 episodes)', fontsize=12)
plt.title('Model Performance Over Time with Different Frame Repeats', fontsize=12)
plt.draw()
input('press [enter] to close.')
plt.close()


#--- CODE TO PLOT one plot
add_to_subplot(plt, './data/model-dtc-fr10-kf4-LSTM.pth', 'blah', 2, 4)
plt.draw()
input('press [enter] to close.')
