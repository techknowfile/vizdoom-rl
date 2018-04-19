from __future__ import print_function

import os
from time import sleep
from random import choice
from vizdoom import *
import argparse

config_file_path = "scenarios/defend_the_center.cfg"

parser = argparse.ArgumentParser()
parser.add_argument('ep')
args = parser.parse_args()

print("Initializing doom...")
game = DoomGame()
game.set_mode(Mode.SPECTATOR)
game.load_config(config_file_path)
game.set_window_visible(True)
game.init()
print("Doom initialized.")

game.replay_episode("./recordings/ep_{}.lmp".format(args.ep))

while not game.is_episode_finished():
    s = game.get_state()

    # Use advance_action instead of make_action.
    game.advance_action()
    sleep(0.03)

    r = game.get_last_reward()
    # game.get_last_action is not supported and don't work for replay at the moment.

    print("State #" + str(s.number))
    print("Game variables:", s.game_variables[0])
    print("Reward:", r)
    print("=====================")

    print("Episode finished.")
    print("total reward:", game.get_total_reward())
    print("************************")

