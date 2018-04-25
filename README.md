# Exploration of DQN in ViZDoom

# Overview
The goal of this project is to analyze deep reinforcement learning (deep RL) and the various architectures
and approaches in Deep RL, as applied to two ViZDoom environments. In particular, we look at the
deep Q-learning algorithm (DQN).

# Dependencies

(1) TensorFlow. Tested on GPU version: https://anaconda.org/anaconda/tensorflow-gpu
CPU version here: https://anaconda.org/conda-forge/tensorflow

(2) Keras. Tested on GPU version: https://anaconda.org/anaconda/keras-gpu)
CPU version here: https://anaconda.org/conda-forge/keras

(3) ViZDoom. Can be installed from here: https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#pypi

(3) Python 2.7

(4) Ubuntu

# How to Run?
There are multiple run time options, however, the recommended option is to leave the settings as they are and
run the code. This will train and test a DQN agent on the ViZDoom 'Basic' environment. Even with a CPU,
the current settings should allow for a reasonable training time in order to view a successfully trained
agent.

# Advanced Run Time Options
In order to test the 'Defend the Center' ViZDoom environment, set the variable 'game_type' in the
main.py file to 'dtc'. Other run time options are primarily for collecting data and not detailed.