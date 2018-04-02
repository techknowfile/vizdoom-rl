from vizdoom import *
import os
import itertools as it
import keras
from random import sample, randint, random
import random
from time import time, sleep
from keras.models import load_model as lm
from keras.layers import Dense, Conv2D, Input, Flatten,LSTM,TimeDistributed
from keras.optimizers import Adam,RMSprop
import numpy as np
import skimage.color, skimage.transform
from tqdm import trange

# Q-learning hyperparams
learning_rate = 0.001
discount_factor = 0.99
epochs = 10
learning_steps_per_epoch = 10000
replay_memory_size = 10000#was 10k
test_memory_size = 1000


# NN learning hyperparam
batch_size = 32

# Training regime
test_episodes_per_epoch = 100

# Image params
resolution = (30, 45)

# Other parameters
frame_repeat = 10
resolution = [30, 45]
kframes = 4
#todo NO CHANNELS, only b/w and k-frames are the number of channels
episodes_to_watch = 10

model_savefile = "models/modelk_frameCnn-fr{}-kf{}.pth".format(frame_repeat, kframes)
if not os.path.exists('models'):
    os.makedirs('models')

save_model = True
load_model = False
skip_learning = False

config_file_path = "../ViZDoom/scenarios/basic.cfg"

import warnings
warnings.filterwarnings("ignore")

def preprocess(img):
    """"""
    img = skimage.transform.resize(img, [resolution[0], resolution[1] ])
    img = img.astype(np.float32)
    return img


class ReplayMemory:
    def __init__(self, capacity):
        state_shape = (capacity, resolution[0], resolution[1])
        test_state_shape = (test_memory_size, resolution[0], resolution[1])
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.test_buffer = np.zeros(test_state_shape , dtype=np.float32)
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
        return ret_buffer


    def get_sample(self, sample_size):


        if self.size - kframes + 1 >= sample_size:
            #note the self.pos is already 1 ahead of the last data entry
            base_idx = self.pos - sample_size - kframes + 1
            #todo, WELL THIS IS DONE. index could be -ve eg: -3. a simple trick. Just allow NEGATIVE INDICES! already auto done
            #if you do a[range(-3,5)] then you get an array of values at indices -3->-1, and 0->5
            samples_kframes_container = []
            samples_action_container = []  # todo, this could be filled quickly with just a list of indices.all contiguous
            samples_s2_container = []
            samples_isTerminal_container = []
            samples_reward_container= []
            for i in range(sample_size):#sample size is not kframes, but could be 32 or 64 (batch size)
                curr_idx = base_idx + i #this will wrap around with negative numbers, so -2, -1,0,1,2 :-)
                frame_indices = range(curr_idx, curr_idx+kframes)
                samples_kframes_container.append(self.s1[frame_indices])
                samples_action_container.append(self.a[curr_idx+kframes-1])
                samples_s2_container.append(self.s2[frame_indices])
                samples_isTerminal_container.append(self.isterminal[curr_idx+kframes-1])
                samples_reward_container.append(self.r[curr_idx+kframes-1])
            return np.array(samples_kframes_container),np.array(samples_action_container), \
                   np.array(samples_s2_container),np.array(samples_isTerminal_container),np.array(samples_reward_container)
        else:
            return None, None, None,None,None #laieeek...literalyeeee....none...Omg.
        #--end outer if




def create_model(available_actions_count):

    state_input = Input((kframes,resolution[0], resolution[1]))

    conv1 = Conv2D(8, 6, strides=3, activation='relu', data_format="channels_first")(
        state_input)  # filters, kernal_size, stride
    conv2 = Conv2D(8, 3, strides=2, activation='relu', data_format="channels_first")(
        conv1)  # filters, kernal_size, stride
    flatten = Flatten()(conv2)

    fc1 = Dense(128,activation='relu')(flatten)
    fc2 = Dense(available_actions_count, activation='relu')(fc1)

    model = keras.models.Model(input=state_input, output=fc2)
    adam = Adam(lr= learning_rate)
    model.compile(loss="mse", optimizer=adam)
    print(model.summary())

    return state_input, model



def learn_from_memory(model):
    """ Use replay memory to learn. Ignore s2 if s1 is terminal """

    if memory.size - kframes + 1 >= batch_size:
        s1, a, s2, isterminal, r = memory.get_sample(batch_size)
        if s1==None:
            return

        q = model.predict(s2, batch_size=batch_size)
        q2 = np.max(q, axis=1)
        target_q = model.predict(s1, batch_size=batch_size)
        target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1 - isterminal) * q2#target_q.shape[0] = batch_size
        model.fit(s1, target_q, verbose=0, shuffle=True)



def get_best_action(state):
    # query memory to get the k-frames
    state_frames = memory.get_test_sample()
    state_frames = state_frames.reshape([1,kframes,resolution[0],resolution[1]])
    q = model.predict(state_frames, batch_size=1)
    m = np.argmax(q, axis=1)[0]
    action = m
    return action


def perform_learning_step(epoch):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        start_eps = 0.4
        end_eps = 0.1
        const_eps_epochs = 0.1 * epochs  # 10% of learning time
        eps_decay_epochs = 0.6 * epochs  # 60% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return 0
            # return start_eps - (epoch - const_eps_epochs) / \
            #        (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return 0
            # return end_eps

    s1 = preprocess(game.get_state().screen_buffer)
    memory.add_to_test_buffer(s1)#we dont add s2 , the next iteration of this loop does that!!

    # With probability eps make a random action.
    eps = exploration_rate(epoch)
    if random.random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        # Choose the best action according to the network.
        #DONT add to test buffer, already done
        a = get_best_action(s1 )
    reward = game.make_action(actions[a], frame_repeat)

    isterminal = game.is_episode_finished()
    s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None

    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, isterminal, reward)
    # IMPORTANT we dont add s2 TO THE TEST BUFFER, the next iteration of this loop does that!!

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
    args, extras = parser.parse_known_args()
    if args.kframes:
        kframes = args.kframes
    if args.test:
        load_model = True
        skip_learning = True
    if args.frame_repeat:
        frame_repeat = args.frame_repeat

    print(("Testing" if skip_learning else "Training"), 'KFrames:', kframes, 'Frame Repeat:', frame_repeat)

    # Create Doom instance
    game = initialize_vizdoom(config_file_path)

    # Action = which buttons are pressed
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Create replay memory which will store the transitions
    memory = ReplayMemory(capacity=replay_memory_size)

    if load_model and os.path.isfile(model_savefile):
        print("Loading model from: ", model_savefile)
        model = lm(model_savefile)
    else:
        my_input, model = create_model(len(actions))

    print("\nTesting...")
    test_episode = []
    test_scores = []
    for test_episode in trange(test_episodes_per_epoch, leave=False):
        memory.reset_test_buffer()
        game.new_episode()
        while not game.is_episode_finished():
            frame = preprocess(game.get_state().screen_buffer)
            memory.add_to_test_buffer(frame)
            best_action_index = get_best_action(frame)
            game.make_action(actions[best_action_index], frame_repeat)
        r = game.get_total_reward()
        test_scores.append(r)

    test_scores = np.array(test_scores)
    print("Results: mean: %.1f +/- %.1f," % (
        test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
          "max: %.1f" % test_scores.max())


    print("Starting the training!")
    time_start = time()
    if not skip_learning:
        for epoch in range(epochs):
            print("\nEpoch %d\n-------" % (epoch + 1))
            train_episodes_finished = 0
            train_scores = []
            memory.reset_test_buffer()

            print("Training...")
            game.new_episode()
            for learning_step in trange(learning_steps_per_epoch, leave=True):
                perform_learning_step(epoch)
                if game.is_episode_finished():
                    score = game.get_total_reward()
                    train_scores.append(score)
                    game.new_episode()
                    train_episodes_finished += 1
                    memory.reset_test_buffer()

            print("%d training episodes played." % train_episodes_finished)

            train_scores = np.array(train_scores)

            print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()), \
                  "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

            print("\nTesting...")
            test_episode = []
            test_scores = []
            for test_episode in trange(test_episodes_per_epoch, leave=False):
                memory.reset_test_buffer()
                game.new_episode()
                while not game.is_episode_finished():
                    frame = preprocess(game.get_state().screen_buffer)
                    memory.add_to_test_buffer(frame)
                    best_action_index = get_best_action(frame)
                    game.make_action(actions[best_action_index], frame_repeat)
                r = game.get_total_reward()
                test_scores.append(r)

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
    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            frame = preprocess(game.get_state().screen_buffer)
            memory.add_to_test_buffer(frame)
            best_action_index = get_best_action(frame)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)

