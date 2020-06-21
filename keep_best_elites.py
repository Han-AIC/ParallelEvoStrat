import os
import shutil
import re
import numpy as np
import torch
from operator import itemgetter
import sys

BEST_OF_THE_BEST = int(sys.argv[1])
solutions = []

for solution_file_name in os.listdir('./elites/'):
    solution_score = re.split("[. _]", solution_file_name)[1]
    solutions.append((solution_file_name, int(solution_score)))

sorted_member_performances = sorted(solutions,
                                   key=itemgetter(1),
                                   reverse=True)
accepted = [x[0] for x in sorted_member_performances[0:BEST_OF_THE_BEST]]

best_score_in_generation = sorted_member_performances[0][1]
best_solution_in_generation = sorted_member_performances[0][0]

if len(list(os.listdir('./best/'))) == 0:
    shutil.copyfile('./elites/' + best_solution_in_generation, './best/' + best_solution_in_generation)
else:
    for best_file_name in os.listdir('./best/'):
        best_score = re.split("[. _]", best_file_name)[1]
        print(best_score)
        print(best_score_in_generation)
        if int(best_score_in_generation) > int(best_score):
            os.remove('./best/' + best_file_name)
            shutil.copyfile('./elites/' + best_solution_in_generation, './best/' + best_solution_in_generation)

for x in sorted_member_performances:
    if x[0] not in accepted:
        os.remove('./elites/' + x[0])
