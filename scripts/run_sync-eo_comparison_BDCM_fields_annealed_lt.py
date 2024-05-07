
from algorithms.sync_eo import run_sim
from algorithms.utils import dump_pickle, RESULT_DIR  
from tqdm import tqdm
import numpy as np
from datetime import datetime

n = 100
hs = np.linspace(0,0.5,25)[1:]
hs = []
extras = [0.3,0.15]
hs = np.concatenate((hs, extras))
hs = np.arange(0,0.5,step=0.01)[1:]
hs = extras
t_wanted = 7


save_every = 50
trials = 1000

results_dir = RESULT_DIR / f'sync-eo-comparison_BDCM-{t_wanted=}-n={n}-annealed_lt' 
results_dir.mkdir(exist_ok=True, parents=True)

saving_seed = np.random.randint(0,100000)
current_date = datetime.now().strftime("%Y-%m-%d")
p = results_dir / f'{current_date}_{saving_seed}.pkl'

pbar = tqdm(total=trials*2)
results = []


for _ in range(trials):
    for c in [1]:#,2]:
        
        for h in np.random.permutation(hs):
            converged = False
            
            while not converged:
                res,_ = run_sim(n=n,h=h,record_energy=True,record_fields=True,max_t=t_wanted+c)
                converged = (res['c'] == c) and (res['transient'] <= t_wanted)
                
            if h not in extras:
                del res['fields']
                del res['xfields']
            results.append(res)
            
            if len(results) % save_every == 0:
                dump_pickle(results, p)
        pbar.update(1)
        