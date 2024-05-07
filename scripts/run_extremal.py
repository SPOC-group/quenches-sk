import matplotlib.pyplot as plt
import numpy as np
import algorithms.extremal as rs
from algorithms.utils import *
from tqdm import tqdm

ns = np.array([40000,29213 ,21335,15581,11180,8310,6086,4432,3237,2364,1726,1261,920,672,491,358,262,191,139,102,74 ,])

results_dir = RESULT_DIR / f'extremal_2'
results_dir.mkdir(exist_ok=True, parents=True)

seed = np.random.randint(0,100000)
trials = int(10000/16)
for n in ns: 
    i = 0
    res = []
    p = results_dir / f'{seed}_{n=}.pkl'
    pbar = tqdm(total=trials)
    while i <= trials:
        i += 1        
        t,n, e, h_min = rs.run_sim(n)
        res.append({'n':n, 'e':e, 't':t, 'h_min':h_min})
        pbar.update(1)
    dump_pickle(res, p)
