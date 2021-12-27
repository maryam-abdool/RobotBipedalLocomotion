#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 02:55:54 2021

@author: maryam
"""

#from __future__ import print_function

import os
import pickle
import neat
import gym
import numpy as np

#from cart_pole import CartPole, discrete_actuator_force
#from movie import make_movie

# load the winner
with open('winner-feedforward8', 'rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward.properties')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

net = neat.nn.FeedForwardNetwork.create(c, config)

env = gym.make('Walker2d-v2')
observation = env.reset()

done = False
for _ in range(30000000):
   #while not done:
    action = net.activate(observation)
    observation, reward, done, info = env.step(action)
    
    env.render()


