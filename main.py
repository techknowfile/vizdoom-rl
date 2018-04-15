from vizdoom import *
from tqdm import trange
from time import time, sleep
from keras.models import load_model as lm
import itertools as it
import os
import numpy as np
import warnings
import matplotlib.pyplot as plt

from Model import NeuralModel
from DQN import ReplayMemory
from DQN import DQN

warnings.filterwarnings("ignore")


class PlayGame:
    def __init__(self, load_model, game_type, model_savefile):
        # Q-learning hyperparams
        self.kframes = 3
        self.learning_rate = 0.0006
        self.discount_factor = 1.0
        self.epochs = 20
        self.learning_steps_per_epoch = 1000
        self.replay_memory_size = 10000
        self.test_memory_size = 10000

        # NN learning hyperparams
        self.batch_size = 64
        self.model_type = 4

        # Training regime
        self.test_episodes_per_epoch = 10

        # Other parameters
        self.frame_repeat = 6
        self.resolution = [30, 45]
        self.episodes_to_watch = 10

        self.model_savefile = model_savefile

        if game_type == 'dtc':
            self.config_file_path = "scenarios/defend_the_center.cfg"
            self.living_reward = 1
            self.default_living_reward = 1
        elif game_type == 'basic':
            self.config_file_path = "scenarios/simpler_basic.cfg"
            self.living_reward = -1
            self.default_living_reward = -1

        self.game_type = game_type
        self.load_model = load_model

        self.model = None
        self.game = None
        self.actions = None
        self.memory = None
        self.dqn = None
        self.current_parameter = None
        self.current_parameter_value = None

        self.initialize_game()

    def initialize_game(self):
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

        self.dqn = DQN(self.memory, self.model, self.game, self.actions, self.resolution, self.frame_repeat, self.batch_size, self.kframes, self.epochs, self.discount_factor, self.model_type)

    # Creates and initializes ViZDoom environment.
    def initialize_vizdoom(self):
        print("Initializing doom...")
        game = DoomGame()
        game.load_config(self.config_file_path)
        game.set_window_visible(False)
        game.set_mode(Mode.PLAYER)
        game.set_screen_format(ScreenFormat.GRAY8)
        game.set_screen_resolution(ScreenResolution.RES_640X480)
        game.set_living_reward(self.living_reward)
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
        self.initialize_game()

        # Train model in new environment
        self.train_agent("dummy_savefile")

        # View model in new environment
        self.view_agent()

    def optimize_hyperprameters(self):
        discount_factor_list = [1, 0.98, 0.95, 0.9]
        model_type = [1, 2, 3, 4]
        reward_shape = [-1, 0, 1]

        hyperparameter_names = ['RS', 'DF', 'MT']
        hyperparameters = [reward_shape, discount_factor_list, model_type]

        for index, hyperparameter_list in enumerate(hyperparameters):
            plt.close(1)
            plt.close(2)
            for parameter in hyperparameter_list:

                self.current_parameter = hyperparameter_names[index]
                self.current_parameter_value = parameter

                if hyperparameter_names[index] == 'RS':
                    self.living_reward = parameter
                    self.discount_factor = 1.0
                    self.model_type = 1
                elif hyperparameter_names[index] == 'DF':
                    self.discount_factor = parameter
                    self.living_reward = self.default_living_reward
                    self.model_type = 1
                elif hyperparameter_names[index] == 'MT':
                    self.model_type = parameter
                    self.living_reward = self.default_living_reward
                    self.discount_factor = 1.0

                # Re-initialize
                self.initialize_game()

                mean_scores, std_scores = self.train_agent()

                # Store data
                self.save_data(mean_scores, std_scores)

    def view_agent(self):
        # Reinitialize the game with window visible
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

    def train_agent(self, model_savefile):
        mean_score_list = []
        std_score_list = []

        print("Starting the training!")
        time_start = time()
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
            mean_score = test_scores.mean()
            std_score = test_scores.std()
            mean_score_list.append(mean_score)
            std_score_list.append(std_score)

            print("Results: mean: %.1f +/- %.1f," % (
                mean_score, std_score), "min: %.1f" % test_scores.min(),
                  "max: %.1f" % test_scores.max())

            print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

        self.game.close()
        print("======================================")
        print("Training finished.")
        self.save_model(model_savefile)

        return mean_score_list, std_score_list

    def save_data(self, mean_scores, std_scores):
        model_directory = 'data/{}/{}/{}/'.format(self.game_type, self.current_parameter, self.current_parameter_value)
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        self.plot_loss(model_directory, mean_scores, std_scores)
        self.save_model(model_directory)
        self.save_parameters(model_directory)

    def save_model(self, directory):
        model_path = '{}model.pth'.format(directory)
        print("Saving the network weights to:", )
        self.model.save(model_path)

    def plot_loss(self, directory, mean_scores, std_scores):
        x_data = list(range(1, self.epochs + 1))
        # Score data
        plt.figure(1)
        plt.plot(x_data, mean_scores, label="{}={}".format(self.current_parameter, self.current_parameter_value), marker='o')
        plt.grid("on")
        plt.xlabel("Epoch")
        plt.ylabel("Avg. Score")
        plt.legend(loc="best", fontsize='x-small')
        plt.title('{} Mean Score vs Epoch'.format(self.current_parameter))

        file_location = '{}{}'.format(directory, 'score_data.png')
        plt.savefig(file_location)

        # STD Data
        plt.figure(2)
        plt.plot(x_data, std_scores, label="STD. {}={}".format(self.current_parameter, self.current_parameter_value), marker='o')
        plt.grid("on")
        plt.xlabel("Epoch")
        plt.ylabel("Score STD")
        plt.legend(loc="best", fontsize='x-small')
        plt.title('{} STD vs Epoch'.format(self.current_parameter))

        file_location = '{}{}'.format(directory, 'std_data.png')
        plt.savefig(file_location)

        # Write data to file
        file_location = '{}{}'.format(directory, 'data.txt')
        text_file = open(file_location, "w")
        text_file.write("Epochs: {}\n".format(x_data))
        text_file.write("Mean Scores: {}\n".format(mean_scores))
        text_file.write("Score STD: {}\n".format(std_scores))
        text_file.close()

    def save_parameters(self, directory):
        file_location = '{}{}'.format(directory, 'parameters.txt')
        text_file = open(file_location, "w")
        text_file.write("Kframes: {}\n".format(self.kframes))
        text_file.write("Learning Rate: {}\n".format(self.learning_rate))
        text_file.write("Discount Factor: {}\n".format(self.discount_factor))
        text_file.write("Epochs: {}\n".format(self.epochs))
        text_file.write("Learning Steps Per Epoch: {}\n".format(self.learning_steps_per_epoch))
        text_file.write("Replay Memory Size: {}\n".format(self.replay_memory_size))
        text_file.write("Test Memory Size: {}\n".format(self.test_memory_size))
        text_file.write("Batch Size: {}\n".format(self.batch_size))
        text_file.write("Model Type: {}\n".format(self.model_type))
        text_file.write("Test Episodes Per Epoch: {}\n".format(self.test_episodes_per_epoch))
        text_file.write("Frame Repeat: {}\n".format(self.frame_repeat))
        text_file.write("Game Type: {}\n".format(self.game_type))
        text_file.close()


def main():
    # 1 = train agent, 2 = test agent, 3 = transfer learning, 4 = test hyperparameters
    option = 1

    # Set to True for option 2 or 3
    load_model = False
    model_savefile = "models/lstm.pth"

    # Other option is 'basic'
    game_type = 'basic'

    pg = PlayGame(load_model, game_type, model_savefile)

    # Train agent
    if option == 1:
        pg.train_agent(model_savefile)
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