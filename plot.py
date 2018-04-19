from matplotlib import pyplot as plt
plt.style.use('ggplot')
import json
import os, glob

results = [] 
for experiment in glob.glob("./data/*"):
    with open(experiment, 'r') as f:
        data = json.load(f)
        data['experiment'] = experiment
        results.append(data)

for result in results:
    print(result['means'])
    plt.plot(result['learning_steps'], result['means'], label=result['experiment'])
    plt.show()


