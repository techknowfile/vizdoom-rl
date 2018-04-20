from vizdoom import *
import math
import json
import os
import itertools as it
import keras
from random import sample, randint, random
import random
from time import time, sleep
from keras.models import Sequential, Model, load_model as lm
from keras.layers import Dense, Activation, Conv2D, Input, Flatten, TimeDistributed, LSTM
from keras.optimizers import Adam, RMSprop
import numpy as np
import skimage.color, skimage.transform
from tqdm import trange

# Q-learning hyperparams
learning_rate = 0.0006
discount_factor = 1.0
epochs = 20
learning_steps_per_epoch = 1000
replay_memory_size = 100000
test_memory_size = 100000

# NN learning hyperparams
batch_size = 64

# Training regime
test_episodes_per_epoch = 100

# Image params
resolution = (30, 45)

# Other parameters
frame_repeat = 6
resolution = [30, 45]
kframes = 3

#TODO: change kframes implementation
# resolution[1] = resolution[1]*kframes

episodes_to_watch = 10

model_savefile = "models/model-dtc-lstm-fr{}-kf{}.pth".format(frame_repeat, kframes)
model_datafile = "data/model-dtc-lstm-fr{}-kf{}.pth".format(frame_repeat, kframes)

# model_savefile = "model-dtc-fr10-kf3.pth"

if not os.path.exists('models'):
    os.makedirs('models')

if not os.path.exists('data'):
    os.makedirs('data')

save_model = True
load_model = True
skip_learning = False

config_file_path = "scenarios/defend_the_center.cfg"

def preprocess(img):
    """"""
    img = skimage.transform.resize(img, [resolution[0], resolution[1] ])
    img = img.astype(np.float32)
    return img

class ReplayMemory:
    def __init__(self, capacity):
        state_shape = (capacity, resolution[0], resolution[1] )
        test_state_shape = (test_memory_size, resolution[0], resolution[1])
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.test_buffer = np.zeros(test_state_shape, dtype=np.float32)
        self.test_buff_pos = 0
        self.test_size = 0

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, :, :] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        samples_s1_container = []
        samples_action_container = []
        samples_s2_container = []
        samples_isTerminal_container = []
        samples_reward_container= []
        for i in sample(range(0, self.size), sample_size):#sample size is not kframes, but could be 32 or 64 (batch size)
            frame_indices = range(i-kframes+1, i+1)#+1 so that the last data point is considered too.
            #this will wrap around with negative numbers, so -2, -1,0,1,2 :-)

            #todo, this will be bad training. Need zero padding. mimic the test buffer

            s1_data = self.s1[frame_indices]
            action_data = self.a[i]
            s2_data = self.s2[frame_indices]
            isTerminal_data = self.isterminal[i]
            reward_data = self.r[i]

            #IF we have a terminal frame in the PRECEEDING k-frames then we DO NOT take those frames. Rather we
            # #repeat the frame after it
            terminal_offset = 0 #not possible case as you will see
            if True in self.isterminal[range(i-kframes+1, i)]: #dont care if the last one is terminal
                for k in range(1,kframes):#excludes the last index
                    if self.isterminal[i-k] : terminal_offset = k
            #--end outer if
            if terminal_offset != 0: #then we need to repeat some frames
                #0:-offset  = -offset repeated as many times
                num_repeats = kframes - terminal_offset
                tmp_reshaped = np.zeros([1] + list(s1_data[-terminal_offset].shape))  # add a leading dummy dimension
                repeat_frames = np.tile(tmp_reshaped, [num_repeats] + [1] * len(resolution))
                s1_data[0:-terminal_offset] = repeat_frames
                s2_data[0:-terminal_offset] = repeat_frames
                #however, the s2 preceeding frame would be s1, so that can be added
                s2_data[-terminal_offset-1] = s1_data[-terminal_offset]

            samples_s1_container.append(s1_data)
            samples_action_container.append(action_data)
            samples_s2_container.append(s2_data)
            samples_isTerminal_container.append(isTerminal_data)
            samples_reward_container.append(reward_data)

        return np.array(samples_s1_container),np.array(samples_action_container), \
               np.array(samples_s2_container),np.array(samples_isTerminal_container),np.array(samples_reward_container)


    def add_to_test_buffer(self, curr_state):
        self.test_buffer[self.test_buff_pos, :, :] = curr_state
        self.test_buff_pos = (self.test_buff_pos + 1) % self.capacity
        self.test_size = min(self.test_size + 1, self.capacity)

    def reset_test_buffer(self):
        self.test_buffer = np.zeros_like(self.test_buffer, dtype=np.float32)
        self.test_buff_pos = 0
        self.test_size = 0

    def get_test_sample(self):
        #get sample of size kframes, or prepend zeros.
        return_state_shape = (kframes, resolution[0], resolution[1])
        ret_buffer = None
        if self.test_size >= kframes:
            ret_buffer = self.test_buffer[self.test_size-kframes:self.test_size,:,:]
        else:#only fill what we have, and have preceeding zero frames (which was already done)
            ret_buffer = np.zeros(return_state_shape, dtype=np.float32)
            ret_buffer[kframes-self.test_size:,:,:] = self.test_buffer[:self.test_size, :, :]
            # num_repeats = kframes-self.test_size
            # tmp_reshaped = self.test_buffer[0].reshape([1]+list(self.test_buffer[0].shape))#add a leading dummy dimension
            # repeat_frames = np.tile(tmp_reshaped,[num_repeats]+[1]*len(resolution) )
            # ret_buffer[:kframes-self.test_size,:,:] = repeat_frames
        return ret_buffer



def create_model(available_actions_count):

    visual_state_input = Input((1,resolution[0], resolution[1]))
    conv1 = Conv2D(8, 6, strides=3, activation='relu', data_format="channels_first")(visual_state_input)
          # filters, kernal_size, stride
    # conv2 = Conv2D(8, 3, strides=2, activation='relu', data_format="channels_first")(
    #     conv1)  # filters, kernal_size, stride

    flatten = Flatten()(conv1)
    fc1 = Dense(128,activation='relu')(flatten)
    fc2 = Dense(64, activation='relu')(fc1)
    visual_model = keras.models.Model(input=visual_state_input, output=fc2)

    state_input = Input((kframes, 1, resolution[0], resolution[1]))
    td_layer = TimeDistributed(visual_model )(state_input)

    lstm_layer = LSTM(64)(td_layer) #IF return sequences is true, it becomes many to many lstm
    fc3 = Dense(32, activation='relu')(lstm_layer)
    fc4 = Dense(available_actions_count)(fc3)

    model = keras.models.Model(input=state_input, output=fc4)
    das_optimizer= RMSprop(lr= learning_rate)
    model.compile(loss="mse", optimizer=das_optimizer)
    print(model.summary())

    return state_input, model

def learn_from_memory(model):
    """ Use replay memory to learn. Ignore s2 if s1 is terminal """

    if memory.size > batch_size:
        s1, a, s2, isterminal, r = memory.get_sample(batch_size)

        s1 = s1.reshape(list(s1.shape[0:2]) + [1] + list(resolution)) #converting to [ batch*kframes*1channel*width*height ]
        s2 = s2.reshape(list(s2.shape[0:2]) + [1] + list(resolution)) #converting to [ batch*kframes*1channel*width*height ]

        q = model.predict(s2, batch_size=batch_size)
        q2 = np.max(q, axis=1)
        target_q = model.predict(s1, batch_size=batch_size)
        target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1 - isterminal) * q2
        model.fit(s1, target_q, verbose=0)


def get_best_action(state):
    '''
    :param state:
    :return:
    '''
    state = state.reshape(list(state.shape[0:2]) + [1] + list(resolution))  # converting to [ batch*kframes*1channel*width*height ]
    q = model.predict(state, batch_size=1)
    m = np.argmax(q, axis=1)[0]
    action = m  # wrong
    return action


def perform_learning_step(epoch):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    def exploration_rate(epoch):
        """# Define exploration rate change over time"""

        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * epochs  # 10% of learning time
        eps_decay_epochs = 0.6 * epochs  # 60% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) / \
                   (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    s1 = preprocess(game.get_state().screen_buffer)

    # With probability eps make a random action.
    eps = exploration_rate(epoch)
    if random.random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        # Choose the best action according to the network.
        memory.add_to_test_buffer(s1)
        state_kframes = memory.get_test_sample()
        state_kframes = state_kframes.reshape([1,kframes,resolution[0],resolution[1]])
        a = get_best_action(state_kframes)
    reward = game.make_action(actions[a], frame_repeat)

    isterminal = game.is_episode_finished()
    #todo RESET the test buffer ELSEWHERE. not here


    s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None

    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, isterminal, reward)

    learn_from_memory(model)

# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--kframes', type=int)
    parser.add_argument('-t', '--test', action='store_true', default=None)
    parser.add_argument('-r', '--frame_repeat', type=int)
    parser.add_argument('-m', '--model', default=None)
    args, extras = parser.parse_known_args()
    if args.kframes:
        kframes = args.kframes
    if args.test:
        load_model = True
        skip_learning = True
    if args.frame_repeat:
        frame_repeat = args.frame_repeat
    if args.model and load_model:
        model_savefile = "{}".format(args.model)
        model_datafile = "data/{}".format(args.model)
    else:
        model_savefile = "models/model-dtc-fr{}-kf{}.pth".format(frame_repeat, kframes)
        model_datafile = "data/model-dtc-fr{}-kf{}.pth".format(frame_repeat, kframes)

    print(("Testing" if skip_learning else "Training"), 'KFrames:', kframes, 'Frame Repeat:', frame_repeat)

    # Create Doom instance
    game = initialize_vizdoom(config_file_path)

    # Action = which buttons are pressed
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]
    actions = [[1,0,0],[0,1,0],[0,0,1]]

    # Create replay memory which will store the transitions
    memory = ReplayMemory(capacity=replay_memory_size)

    if load_model and os.path.isfile(model_savefile):
        print("Loading model formatom: ", model_savefile)
        model = lm(model_savefile)
    else:
        my_input, model = create_model(len(actions))

    print("Starting the training!")
    time_start = time()
    if not skip_learning:
        for epoch in range(epochs):
            print("\nEpoch %d\n-------" % (epoch + 1))
            train_episodes_finished = 0
            train_scores = []

            print("Training...")
            game.new_episode()
            memory.reset_test_buffer()
            for learning_step in trange(learning_steps_per_epoch, leave=True):
                perform_learning_step(epoch)
                if game.is_episode_finished():
                    memory.reset_test_buffer()
                    score = game.get_total_reward()
                    train_scores.append(score)
                    game.new_episode()
                    train_episodes_finished += 1

            print("%d training episodes played." % train_episodes_finished)

            train_scores = np.array(train_scores)

            print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

            print("\nTesting...")
            test_episode = []
            test_scores = []
            for test_episode in trange(test_episodes_per_epoch, leave=False):
                game.new_episode()
                memory.reset_test_buffer()
                while not game.is_episode_finished():
                    frame = preprocess(game.get_state().screen_buffer)
                    memory.add_to_test_buffer(frame)
                    state_kframes = memory.get_test_sample()
                    state_kframes = state_kframes.reshape([1, kframes, resolution[0], resolution[1]])
                    best_action_index = get_best_action(state_kframes)

                    game.make_action(actions[best_action_index], frame_repeat)
                r = game.get_total_reward()
                test_scores.append(r)
                # --end while
            # --end for
            test_scores = np.array(test_scores)
            print("Results: mean: %.1f +/- %.1f," % (
                test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
                  "max: %.1f" % test_scores.max())

            print("Saving the network weigths to:", model_savefile)
            model.save(model_savefile)

            print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

    game.close()
    print("======================================")
    print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    # game.set_window_visible(True)
    # game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    cleaner = None
    for i in range(episodes_to_watch):
        game.new_episode('./recordings/ep_{}.lmp'.format(i))
        if cleaner:
            os.remove('recordings/ep_{}.lmp'.format(cleaner))
            cleaner = None
        while not game.is_episode_finished():
            frame = preprocess(game.get_state().screen_buffer)
            memory.add_to_test_buffer(frame)
            state_kframes = memory.get_test_sample()
            state_kframes = state_kframes.reshape([1, kframes, resolution[0], resolution[1]])  # 1 is the batch size
            best_action_index = get_best_action(state_kframes)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        # sleep(1.0)
        score = game.get_total_reward()
        if score < 24:
            cleaner = i
        print(i, "Total score: ", score)


