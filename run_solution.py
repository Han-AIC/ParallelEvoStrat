import json
import numpy as np
import torch
from model import Model
from environment import Environment

environments = {0: 'CartPole-v1', 1: 'LunarLander-v2', 2: 'HumanoidBulletEnv-v0'}
ENV_NAME = environments[1]
SOLUTION_SCORE = 227
MAX_STEPS = 10000

solution_path = "./solutions/" + ENV_NAME + '_' + str(SOLUTION_SCORE) + "/solution.pth"
params_path = "./solutions/" + ENV_NAME + '_' + str(SOLUTION_SCORE) + "/model_params.json"

with open(params_path) as json_file:
    experiment_params = json.load(json_file)

solution_model_structure = experiment_params['model_structure']
# conv = experiment_params['conv']
render = True

solution_model = Model(solution_model_structure)
solution_model.load_state_dict(torch.load(solution_path), strict=False)

# environment = Environment(solution_model, ENV_NAME, experiment_params['action_shape'], experiment_params['pybullet'], conv, render)
environment = Environment(solution_model, ENV_NAME, 4, False, False, False)

state = environment.reset()
action = environment.select_action_from_policy(state)

for i in range(MAX_STEPS):
    environment.render_env()
    next_state, reward, done, info = environment.step(action)
    next_action = environment.select_action_from_policy(next_state)
    action = next_action
    state = next_state
    if done:
        break
