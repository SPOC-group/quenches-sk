
from algorithms.sync_rel import run_sim
from algorithms.utils import dump_pickle, RESULT_DIR  
from tqdm import tqdm
import numpy as np
from datetime import datetime

ns = [100,200,400,1000,2000,4000]#,10000,20000]
hs = 1/np.linspace(0.1,2.5,30)
save_every = 10
trials = 124

results_dir = RESULT_DIR / 'sync-rel-vary-h-a100' 
results_dir.mkdir(exist_ok=True, parents=True)

saving_seed = np.random.randint(0,100000)
current_date = datetime.now().strftime("%Y-%m-%d")
p = results_dir / f'{current_date}_{saving_seed}.pkl'

pbar = tqdm(total=len(ns)*trials)
results = []
for _ in range(trials):
    for h in np.random.permutation(hs):
        for n in np.random.permutation(ns):
            results.append(run_sim(n=n,h=h,max_t=100))
            pbar.update(1)
            if len(results) % save_every == 0:
                dump_pickle(results, p)
    