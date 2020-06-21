
import os
import torch
from model import Model
from helpers import *
import json
from collections import defaultdict
import numpy as np


if len([name for name in os.listdir('./elites')]) != 0:
    with open('./params/model_params.json') as json_file:
        experiment_params = json.load(json_file)
    model_structure = experiment_params['model_structure']
    lr = experiment_params['lr']

    previous_elites = defaultdict()
    for idx, elite_file_name in enumerate(os.listdir('./elites')):
        elite_state = torch.load('./elites/' + elite_file_name)
        previous_elites[idx] = elite_state

    previous_means = calculate_means(previous_elites, model_structure)
    if len(list(os.listdir(('./running_mean/')))) == 0:
        torch.save(previous_means, './running_mean/running_means.pth')
    else:
        running_means = torch.load('./running_mean/running_means.pth')
        for layer in running_means:
            running_means[layer] += lr * (previous_means[layer] - running_means[layer])
        torch.save(previous_means, './running_mean/running_means.pth')
