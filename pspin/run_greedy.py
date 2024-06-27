import matplotlib.pyplot as plt
import numpy as np
from algorithms.all import *
from algorithms.utils import *
from tqdm import tqdm

ns = np.array(list(reversed([491,358,262,191,139,102,74,50,25,10]))) # 1261,920,672
ns=[1726]
results_dir = RESULT_DIR / f'greedy'
results_dir.mkdir(exist_ok=True, parents=True)

res = []
trials = 1000

# use twdm progress bar
seed = np.random.randint(0,100000)
for n in ns: 
    i = 0
    print(n)
    res = []
    p = results_dir / f'{seed}_{n=}.pkl'
    pbar = tqdm(total=trials)
    while i <= trials:
        i += 1
        
        n, e, _, t = run_greedy(n)
        res.append({'n':n, 'e':e, 't':t, })

        pbar.update(1)
        dump_pickle(res, p)
