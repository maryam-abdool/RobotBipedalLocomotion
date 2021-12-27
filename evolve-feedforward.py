#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 20:21:28 2021

@author: maryam
"""
import multiprocessing
import os
import pickle

import neat

#import cart_pole
#import visualize

import gym
import numpy as np

from timeit import default_timer as timer
from datetime import timedelta

runs_per_net = 2
#simulation_seconds = 60.0

from keras import backend as K
import tensorflow.compat.v1 as tf

NUM_PARALLEL_EXEC_UNITS = 2
config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=2,
                       allow_soft_placement=True, device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})
session = tf.Session(config=config)
K.set_session(session)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []
    
    for runs in range(runs_per_net):

        env = gym.make('Walker2d-v2')
        observation = env.reset()

        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        done = False
        while not done:
            action = net.activate(observation)
            observation, reward, done, info = env.step(action)
            fitness += reward

        fitnesses.append(fitness)

    # The genome's fitness is its worst performance across all runs.
    return np.max(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    #tensorflow
    #tf.compat.v1.disable_eager_execution()
    tf.disable_v2_behavior()
    '''NUM_PARALLEL_EXEC_UNITS = 2
    config = tf.ConfigProto(intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS, inter_op_parallelism_threads=2,
                           allow_soft_placement=True, device_count={'CPU': NUM_PARALLEL_EXEC_UNITS})
    session = tf.Session(config=config)
    K.set_session(session)
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["KMP_BLOCKTIME"] = "30"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"'''
    # Load the config file, which is assumed to live in
    # the same directory as this script.

    #timer
    start = timer()
    #print(start);
    
    
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.properties')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate)

    # Save the winner.
    with open('winner-feedforward8', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)


    end = timer()
    sec = end - start
    print(sec)
    print(timedelta(seconds=sec))


    '''visualize.plot_stats(stats, ylog=True, view=True, filename="feedforward-fitness.svg")
    visualize.plot_species(stats, view=True, filename="feedforward-speciation.svg")

    node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    visualize.draw_net(config, winner, True, node_names=node_names)

    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward.gv")
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward-enabled.gv", show_disabled=False)
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward-enabled-pruned.gv", show_disabled=False, prune_unused=True)'''


if __name__ == '__main__':
    run()