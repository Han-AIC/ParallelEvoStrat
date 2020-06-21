import torch
import numpy as np
from model import Model

def calculate_means(pop_dict, model_structure):
    means = prep_base_state_dict(model_structure)
    weighting = 1 / len(pop_dict.keys())
    for elite_idx in pop_dict:
        elite_state = pop_dict[elite_idx]
        for layer in means:
            means[layer] += weighting * elite_state[layer]
    return means

def prep_base_state_dict(model_structure):
    template = Model(model_structure)
    base_state_dict = template.state_dict()
    for layer in base_state_dict:
        zeroes = torch.from_numpy(np.zeros(base_state_dict[layer].shape))
        base_state_dict.update({layer: zeroes})
    return base_state_dict
