import os
import shutil
import re
import numpy as np
import torch

best = np.NINF
best_solution_name = None
for solution_file_name in os.listdir('./elites/'):
    solution_score = re.split("[. _]", solution_file_name)[1]
    if int(solution_score) > best:
        best = int(solution_score)
        best_solution_name = solution_file_name

SOLUTION_PATH = './solutions/' + best_solution_name.split('.')[0]
if not os.path.exists(SOLUTION_PATH):
    os.makedirs(SOLUTION_PATH)

torch.save(torch.load('./elites/' + best_solution_name), SOLUTION_PATH + '/solution.pth')
shutil.move('./params/model_params.json', SOLUTION_PATH + '/model_params.json')
