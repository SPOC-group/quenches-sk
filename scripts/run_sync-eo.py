
from algorithms.sync_eo import run_sim
from algorithms.utils import dump_pickle, RESULT_DIR  
from tqdm import tqdm
import numpy as np
from datetime import datetime

ns = [100,400,1000,2000,4000,10000,20000] # 100
hs = np.linspace(-0.6,0.0,90)
save_every = 10000
trials = 124

results_dir = RESULT_DIR / 'sync-eo-vary-h' 
results_dir.mkdir(exist_ok=True, parents=True)

saving_seed = np.random.randint(0,100000)
current_date = datetime.now().strftime("%Y-%m-%d")
p = results_dir / f'{current_date}_{saving_seed}.pkl'

pbar = tqdm(total=len(ns)*trials*len(hs))
results = []
for _ in range(trials):
    for h in np.random.permutation(hs):
        for n in np.random.permutation(ns):
            results.append(run_sim(n=n,h=h)[0])
            pbar.update(1)
dump_pickle(results, p)
    