import matplotlib.pyplot as plt
import numpy as np
import algorithms.extremal as rs
from algorithms.utils import *
from tqdm import tqdm

ns = np.array([100,200,400,800,1600,3200])

results_dir = RESULT_DIR / f'extremal_cnt_flips'
results_dir.mkdir(exist_ok=True, parents=True)

seed = np.random.randint(0,100000)
trials = 400
for n in ns: 
    i = 0
    print(n)
    res = []
    p = results_dir / f'{seed}_{n=}.pkl'
    pbar = tqdm(total=trials)
    while i <= trials:
        i += 1
        
        t,n, e, h_min, flips = rs.run_sim(n,track_flips=True)
        res.append({'n':n, 'e':e, 't':t, 'h_min':h_min, 'flips':flips})

        pbar.update(1)
    dump_pickle(res, p)

