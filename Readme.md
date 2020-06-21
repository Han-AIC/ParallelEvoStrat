1. When main is called, it searches elites folder for existing solution .pth files.
2. If none exist, it spawns initial generation as normal.
3. If there are solution .pth files, load them, take the mean. Use it to generate a new population.
4. Evaluate single generation and select elites as normal.
5. Save best performer of each generation for each thread in a special holding folder. 
