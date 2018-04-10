from vizdoom import *
from tqdm import trange
from time import time, sleep
from keras.models import load_model as lm
import itertools as it
import os
import numpy as np
import argparse
import warnings

from Model import NeuralModel
from DQN import ReplayMemory
from DQN import DQN

warnings.filterwarnings("ignore")


class PlayGame:
    def __init__(self, kframes, skip_learning, load_model, game_type):
        # Q-learning hyperparams
        self.kframes = kframes
        self.skip_learning = skip_learning
        self.learning_rate = 0.001
        self.discount_factor = 1.0
        self.epochs = 1
        self.learning_steps_per_epoch = 1000
        self.replay_memory_size = 10000
        self.test_memory_size = 10000

        # NN learning hyperparams
        self.batch_size = 64
        self.model_type = 3

        # Training regime
        self.test_episodes_per_epoch = 100

        # Other parameters
        self.frame_repeat = 10
        self.resolution = [30, 45]
        self.episodes_to_watch = 10

        self.model_savefile = "models/1.pth"

        if game_type == 'dtc':
            self.config_file_path = "../ViZDoom/scenarios/defend_the_center.cfg"
        elif game_type == 'basic':
            self.config_file_path = "../ViZDoom/scenarios/simpler_basic.cfg"

        self.game_type = game_type
        self.load_model = load_model

        self.model = None
        self.game = None
        self.actions = None
        self.memory = None
        self.dqn = None

        self.initalize_game()

    def initalize_game(self):
        # Create Doom instance
        self.game = self.initialize_vizdoom()

        # Action = which buttons are pressed
        n = self.game.get_available_buttons_size()
        self.actions = [list(a) for a in it.product([0, 1], repeat=n)]
        self.actions = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

        if self.load_model:
            print("Loading model from: ", self.model_savefile)
            self.model = lm(self.model_savefile)
            pass
        else:
            model_class = NeuralModel(len(self.actions), self.learning_rate, self.kframes)
            my_input, self.model = model_class.create_model(self.model_type)

        # Create replay memory which will store the transitions
        self.memory = ReplayMemory(capacity=self.replay_memory_size, kframes=self.kframes, resolution=self.resolution, test_memory_size=self.test_memory_size)

        self.dqn = DQN(self.memory, self.model, self.game, self.actions, self.resolution, self.frame_repeat, self.batch_size, self.kframes, self.epochs, self.discount_factor)

    # Creates and initializes ViZDoom environment.
    def initialize_vizdoom(self):
        print("Initializing doom...")
        game = DoomGame()
        game.load_config(self.config_file_path)
        game.set_window_visible(False)
        game.set_mode(Mode.PLAYER)
        game.set_screen_format(ScreenFormat.GRAY8)
        game.set_screen_resolution(ScreenResolution.RES_640X480)
        game.init()
        print("Doom initialized.")
        return game

    def transfer_learn(self):
        # Train a new agent if we are not loading one
        if not self.load_model:
            # Train model in environment
            self.train_agent()

            # View model in environment
            self.view_agent()

        # Reverse environment
        if self.game_type == 'dtc':
            self.config_file_path = "../ViZDoom/scenarios/simpler_basic.cfg"
        elif self.game_type == 'basic':
            self.config_file_path = "../ViZDoom/scenarios/defend_the_center.cfg"

        # Reinitialize game environment
        self.initalize_game()

        # Train model in new environment
        self.train_agent()

        # View model in new environment
        self.view_agent()

    def optimize_hyperprameters(self):
        k_frame_list = [1, 2, 4, 8]
        learning_rate_list = [0.001, 0.01, 0.05]
        discount_factor_list = [1, 0.98, 0.95, 0.9]
        model_type = [1, 2, 3]

        hyperparameter_names = ['kframes', 'learning_rate', 'discount_factor', 'model_type']
        hyperparameters = [k_frame_list, learning_rate_list, discount_factor_list, model_type]

        for index, hyperparameter_list in enumerate(hyperparameters):
            for parameter in hyperparameter_list:

                if hyperparameter_names[index] == 'kframes':
                    self.kframes = parameter
                elif hyperparameter_names[index] == 'learning_rate':
                    self.learning_rate = parameter
                elif hyperparameter_names[index] == 'discount_factor':
                    self.discount_factor = parameter
                elif hyperparameter_names[index] == 'model_type':
                    self.model_type = parameter

                # Re-initialize
                self.initalize_game()

                self.train_agent()
                #self.view_agent()

    def view_agent(self):
        # Reinitialize the game with window visible
        self.game.close()
        self.game.set_window_visible(True)
        self.game.set_mode(Mode.ASYNC_PLAYER)
        self.game.init()

        for _ in range(self.episodes_to_watch):
            self.game.new_episode()
            while not self.game.is_episode_finished():
                frame = self.dqn.preprocess(self.game.get_state().screen_buffer)
                self.memory.add_to_test_buffer(frame)
                state_kframes = self.memory.get_test_sample()
                state_kframes = state_kframes.reshape([1, self.kframes, self.resolution[0], self.resolution[1]])  # 1 is the batch size
                best_action_index = self.dqn.get_best_action(state_kframes)

                # Instead of make_action(a, frame_repeat) in order to make the animation smooth
                self.game.set_action(self.actions[best_action_index])
                for _ in range(self.frame_repeat):
                    self.game.advance_action()

            # Sleep between episodes
            sleep(1.0)
            score = self.game.get_total_reward()
            print("Total score: ", score)

    def train_agent(self):
        print("Starting the training!")
        time_start = time()
        if not self.skip_learning:
            for epoch in range(self.epochs):
                print("\nEpoch %d\n-------" % (epoch + 1))
                train_episodes_finished = 0
                train_scores = []

                print("Training...")
                self.game.new_episode()
                self.memory.reset_test_buffer()
                for _ in trange(self.learning_steps_per_epoch, leave=True):
                    self.dqn.perform_learning_step(epoch)
                    if self.game.is_episode_finished():
                        self.memory.reset_test_buffer()
                        score = self.game.get_total_reward()
                        train_scores.append(score)
                        self.game.new_episode()
                        train_episodes_finished += 1

                print("%d training episodes played." % train_episodes_finished)

                train_scores = np.array(train_scores)

                print("Results: mean: %.1f +/- %.1f," % (train_scores.mean(), train_scores.std()),
                      "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

                print("\nTesting...")
                test_scores = []
                for _ in trange(self.test_episodes_per_epoch, leave=False):
                    self.game.new_episode()
                    self.memory.reset_test_buffer()
                    while not self.game.is_episode_finished():
                        frame = self.dqn.preprocess(self.game.get_state().screen_buffer)
                        self.memory.add_to_test_buffer(frame)
                        state_kframes = self.memory.get_test_sample()
                        state_kframes = state_kframes.reshape([1, self.kframes, self.resolution[0], self.resolution[1]])
                        best_action_index = self.dqn.get_best_action(state_kframes)

                        self.game.make_action(self.actions[best_action_index], self.frame_repeat)
                    r = self.game.get_total_reward()
                    test_scores.append(r)

                test_scores = np.array(test_scores)
                print("Results: mean: %.1f +/- %.1f," % (
                    test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
                      "max: %.1f" % test_scores.max())

                print("Saving the network weights to:", self.model_savefile)
                self.model.save(self.model_savefile)

                print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

                self.game.close()
        print("======================================")
        print("Training finished.")


def main():
    if not os.path.exists('models'):
        os.makedirs('models')

    # 1 = train agent, 2 = test agent, 3 = transfer learning, 4 = test hyperparameters
    option = 1
    kframes = 2

    save_model = True
    load_model = True
    skip_learning = False

    # Other option is 'basic'
    game_type = 'dtc'

    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--kframes', type=int)
    parser.add_argument('-t', '--test')
    args, extras = parser.parse_known_args()

    if args.kframes:
        kframes = args.kframes
    if args.test:
        load_model = True
        skip_learning = True

    pg = PlayGame(kframes, skip_learning, load_model, game_type)

    # Train agent
    if option == 1:
        pg.train_agent()
        pg.view_agent()
    # Test a trained agent
    elif option == 2:
        pg.view_agent()
    # Transfer agent from one environment to another
    elif option == 3:
        pg.transfer_learn()
    # Search across a range of hyperparameters to optimize them
    elif option == 4:
        pg.optimize_hyperprameters()


if __name__ == '__main__':
    main()