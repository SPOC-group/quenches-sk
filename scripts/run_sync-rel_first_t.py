
from algorithms.sync_rel import run_sim
from algorithms.utils import dump_pickle, RESULT_DIR  
from tqdm import tqdm
import numpy as np
from datetime import datetime

ns = [4000]#,10000,20000]
hs = np.linspace(0.0,5.0,12)#[0.1, 0.15, 0.2, 0.25, 0.5, 1.0]#np.linspace(0,0.5,30) +  
save_every = 10
trials = 10

results_dir = RESULT_DIR / 'sync-rel-vary-first-t-1000-2' 
results_dir.mkdir(exist_ok=True, parents=True)

saving_seed = np.random.randint(0,100000)
current_date = datetime.now().strftime("%Y-%m-%d")
p = results_dir / f'{current_date}_{saving_seed}.pkl'

pbar = tqdm(total=len(ns)*trials)
results = []
for _ in range(trials):
    for h in np.random.permutation(hs):
        for n in np.random.permutation(ns):
            results.append(run_sim(n=n,h=h,max_t=500, record_energy=True))
            pbar.update(1)
            if len(results) % save_every == 0:
                dump_pickle(results, p)
    