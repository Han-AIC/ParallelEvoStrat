import os
import sys
import gym
import pybullet_envs
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque, OrderedDict, Counter
from operator import itemgetter
import copy
import time
import gc
import json
import copy

from joblib import Parallel, delayed
import multiprocessing

from spawn import Spawner
from environment import Environment
from collections import defaultdict

from tqdm import tqdm, tnrange, tqdm_notebook
import time

from model import Model
from helpers import *

gym.logger.set_level(40)

class EvoStrat_Experiment:

    """
    1. Instantiates an environment for each population member to undergo evaluation.
    2. Keeps track of states over time, selects actions probabilistically from the
       output of each member model.
    3. Steps environment forward using selected action.
    4. Resets environment using a new novel random seed.
    """

    def __init__(self,
                 experiment_params,
                 GENERATION_ID,
                 THREAD_ID):

        self.env_name = experiment_params['env_name']
        self.model_structure = experiment_params['model_structure']

        self.MAX_STEPS = experiment_params['MAX_STEPS']
        self.num_episodes = experiment_params['num_episodes']
        self.population_size = experiment_params['population_size']
        self.elite_proportion = experiment_params['elite_proportion']
        self.step_size = experiment_params['step_size']
        self.lr = experiment_params['lr']
        self.action_shape = experiment_params['action_shape']
        self.pybullet = experiment_params['pybullet']
        self.conv = experiment_params['conv']

        self.spawner = Spawner(self.population_size,
                               self.model_structure)

        self.current_population = defaultdict()
        self.population_elites = defaultdict()
        self.population_performance = defaultdict()

        self.best_performer = {'gen_idx': np.NINF,
                                'member_idx': np.NINF,
                                'performance': np.NINF}

        self.GENERATION_ID = GENERATION_ID
        self.THREAD_ID = THREAD_ID

    def initialize_populations(self):
        num_existing_elites = len([name for name in os.listdir('./elites')])
        if num_existing_elites == 0:
            self.current_population = self.spawner.generate_initial_population()
        else:
            running_means = torch.load('./running_mean/running_means.pth')
            self.current_population = self.spawner.generate_population(running_means, self.step_size)

    def evaluate_one_generation(self):
        pbar_within_generation = tqdm(total=len(self.current_population.keys()), desc='Members', leave=False)
        for member_idx in self.current_population:
            member_performance = self.evaluate_one_member(self.current_population[member_idx])
            self.population_performance[member_idx] = member_performance
            pbar_within_generation.update(1)
        time.sleep(0.001)

    def evaluate_one_member(self,
                            member):
        environment = Environment(member, self.env_name, self.action_shape, self.pybullet, self.conv, False)
        reward_window = deque(maxlen=self.num_episodes)
        for episode_idx in range(1, self.num_episodes+1):
            state = environment.reset()
            action = environment.select_action_from_policy(state)
            reward_per_episode = 0
            for i in range(self.MAX_STEPS):
                next_state, reward, done, info = environment.step(action)
                next_action = environment.select_action_from_policy(next_state)
                action = next_action
                state = next_state
                reward_per_episode += reward
                if done:
                    reward_window.append(reward_per_episode)
                    break
        return np.mean(reward_window)

    def select_top_performers(self):
        member_performances = list(self.population_performance.items())
        sorted_member_performances = sorted(member_performances,
                                       key=itemgetter(1),
                                       reverse=True)
        elites = sorted_member_performances[0:int(len(sorted_member_performances) * self.elite_proportion)]
        for elite in elites:
            self.population_elites[elite[0]] = elite[1]

    def select_best_performer(self):
        for elite in self.population_elites:
            performance = self.population_elites[elite]
            if performance > self.best_performer['performance']:
                self.best_performer = {'member_idx': elite,
                                        'performance': performance}

    def save_best_performer(self):
        model_state = self.current_population[self.best_performer['member_idx']].state_dict()
        SOLUTION_PATH = './intermediate_solutions/' + self.env_name + '_' + str(int(self.best_performer['performance'])) + '_Gen' + self.GENERATION_ID + '_Thrd' + self.THREAD_ID + '.pth'
        torch.save(model_state, SOLUTION_PATH)

    def average_elite_performance(self):
        return np.sum(list(self.population_elites.values())) / len(self.population_elites)

    def average_whole_performance(self):
        return np.sum(list(self.population_performance.values())) / len(self.population_performance)

    def run_experiment(self):
        self.initialize_populations()
        self.evaluate_one_generation()
        self.select_top_performers()
        self.select_best_performer()
        self.save_best_performer()

        avg_whole_population_performance = self.average_whole_performance()
        avg_elite_performance = self.average_elite_performance()

        print("================================================================")
        print(f"Generation: {self.GENERATION_ID} Thread: {self.THREAD_ID}")
        print(f"Average Elite Score: {avg_elite_performance} Average Population Score: {avg_whole_population_performance}")
