import matplotlib.pyplot as plt
import numpy as np
import algorithms.reluctant as rs
from algorithms.utils import *
from tqdm import tqdm

ns = np.array([100,200,400,800,1600,3200])
results_dir = RESULT_DIR / f'reluctant_cnt_flips'
results_dir.mkdir(exist_ok=True, parents=True)

res = []

# use twdm progress bar
seed = np.random.randint(0,100000)
for n in ns: 
    i = 0
    print(n)
    res = []
    p = results_dir / f'{seed}_{n=}.pkl'
    pbar = tqdm(total=int(3500/16/2))
    while i <= 400:
        i += 1
        
        t,n, e, h_min, flips = rs.run_sim(n,track_flips=True)
        res.append({'n':n, 'e':e, 't':t, 'h_min':h_min, 'flips':flips})

        pbar.update(1)
        if i % 10 == 0:
            dump_pickle(res, p)
