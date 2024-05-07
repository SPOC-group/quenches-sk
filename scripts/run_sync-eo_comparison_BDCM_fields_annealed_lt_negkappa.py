
from algorithms.sync_eo import run_sim
from algorithms.utils import dump_pickle, RESULT_DIR  
from tqdm import tqdm
import numpy as np
from datetime import datetime

n = 80
hs = np.arange(-0.1,0.0,step=0.01)[::-1]
extras = []
t_wanted = 7
print(hs)

save_every = 2
trials = 100

results_dir = RESULT_DIR / f'sync-eo-comparison_BDCM-{t_wanted=}-n={n}-annealed_lt_negkappa_2' 
results_dir.mkdir(exist_ok=True, parents=True)

saving_seed = np.random.randint(0,100000)
current_date = datetime.now().strftime("%Y-%m-%d")
p = results_dir / f'{current_date}_{saving_seed}.pkl'

pbar = tqdm(total=trials*2)
results = []


for _ in range(trials):
    for c in [1]:#,2]:
        
        for h in hs:
            converged = False
            trials = 0
            v = None
            while not converged:
                res,_ = run_sim(n=n,h=h,record_energy=True,record_fields=True,max_t=t_wanted+c)
                converged = (res['c'] == c) and (res['transient'] <= t_wanted)
                trials += 1
                
            print(h, trials,trials/2**n)
                
            if h not in extras:
                del res['fields']
                del res['xfields']
            results.append(res)
            
            if len(results) % save_every == 0:
                dump_pickle(results, p)
        pbar.update(1)
        