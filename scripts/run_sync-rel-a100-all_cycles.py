
from algorithms.sync_rel_cycle_check import run_sim
from algorithms.utils import dump_pickle, RESULT_DIR  
from tqdm import tqdm
import numpy as np
from datetime import datetime


ns = [400]#,10000,20000]
hs = [0.02,0.04,0.08]#np.linspace(0.1,10,40) # np.linspace(0.1,20,40)
save_every = 20
trials = 10

results_dir = RESULT_DIR / 'sync-rel-vary-h-a100-cycle' 
results_dir.mkdir(exist_ok=True, parents=True)

saving_seed = np.random.randint(0,100000)
current_date = datetime.now().strftime("%Y-%m-%d")
p = results_dir / f'{current_date}_{saving_seed}.pkl'

pbar = tqdm(total=len(ns)*trials)
results = []
for _ in range(trials):
    for h in np.random.permutation(hs):
        for n in np.random.permutation(ns):
            results.append(run_sim(n=n,h=h,max_t=10000,record_energy=True))
            pbar.update(1)
            if len(results) % save_every == 0:
                dump_pickle(results, p)
dump_pickle(results, p)