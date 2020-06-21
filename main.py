import os
import json
import gym
import pybullet_envs
import sys

from experiment import EvoStrat_Experiment

gym.logger.set_level(60)

ENV = 'HumanoidBulletEnv-v0'
# input_dim = gym.make(ENV).observation_space.shape[0]
# output_dim = gym.make(ENV).action_space.n
# state = gym.make(ENV).reset()
# print(state.reshape(1, -1).shape)
input_dim = gym.make(ENV).observation_space.shape[0]
output_dim = gym.make(ENV).action_space.shape[0]

# print(gym.make(ENV).observation_space.shape)
experiment_params = {
    'env_name': ENV,
    'model_structure': {
                    "fc1":{"layer_size_mapping": {"in_features": input_dim,
                                                 "out_features": 64},
                                         "layer_type": "linear",
                                         "activation": "tanh"},
                    "fc2":{"layer_size_mapping": {"in_features": 64,
                                                 "out_features": 64},
                                         "layer_type": "linear",
                                         "activation": "tanh"},
                    "fc3":{"layer_size_mapping": {"in_features": 64,
                                                 "out_features": 32},
                                         "layer_type": "linear",
                                         "activation": "tanh"},
                    "fc4":{"layer_size_mapping": {"in_features": 32,
                                                 "out_features": 32},
                                         "layer_type": "linear",
                                         "activation": "tanh"},
                    "fc5":{"layer_size_mapping": {"in_features": 32,
                                                 "out_features": 32},
                                         "layer_type": "linear",
                                         "activation": "tanh"},
                    "fc6":{"layer_size_mapping": {"in_features": 32,
                                                 "out_features": 32},
                                         "layer_type": "linear",
                                         "activation": "tanh"},
                    "fc7":{"layer_size_mapping": {"in_features": 32,
                                                 "out_features": 32},
                                         "layer_type": "linear",
                                         "activation": "tanh"},
                    "fc8":{"layer_size_mapping": {"in_features": 32,
                                                 "out_features": 32},
                                         "layer_type": "linear",
                                         "activation": "tanh"},
                    "fc9":{"layer_size_mapping": {"in_features": 32,
                                                 "out_features": output_dim},
                                         "layer_type": "linear",
                                         "activation": "tanh"},

                   },
   'MAX_STEPS': 1000,
   'num_episodes': 10,
   'population_size': 10,
   'elite_proportion': 0.2,
   'step_size': 0.05,
   'lr': 0.1,
   'action_shape': output_dim,
   'pybullet': True,
   'conv': False
}

GENERATION_ID= sys.argv[1]
THREAD_ID = sys.argv[2]

experiment = EvoStrat_Experiment(experiment_params, GENERATION_ID, THREAD_ID)
solution_score = experiment.run_experiment()


if len(list(os.listdir('./params/'))) == 0:
    PARAMS_PATH = './params/model_params.json'
    with open(PARAMS_PATH, 'w') as outfile:
        json.dump(experiment_params, outfile)
