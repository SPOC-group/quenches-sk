
from algorithms.sync_eo import run_sim
from algorithms.utils import dump_pickle, RESULT_DIR  
from tqdm import tqdm
import numpy as np
from datetime import datetime

n = 80
hs = np.arange(-0.1,0.0,step=0.01)[::-1]#[:5]
extras = []
t_wanted = 7

save_every = 2
trials = 200

results_dir = RESULT_DIR / f'sync-eo-comparison_BDCM-{t_wanted=}-n={n}-quenched_lt_negkappa_2' 
results_dir.mkdir(exist_ok=True, parents=True)

saving_seed = np.random.randint(0,100000)
current_date = datetime.now().strftime("%Y-%m-%d")
p = results_dir / f'{current_date}_{saving_seed}.pkl'

pbar = tqdm(total=trials*2)
results = []

t_wanted = 7

for _ in range(trials):
    for c in [1]:#,2]:
        for h in np.random.permutation(hs):
            converged = False
            print(h)
            J_ij = None
            misses = 0
            trials = 0
            v = None#np.random.choice([-1,1],size=n) 
            while not converged:
                res, J_ij = run_sim(n=n,h=h,record_energy=True,record_fields=True,max_t=t_wanted+c,J_ij=J_ij,v=v)
                converged = (res['c'] == c) and (res['transient'] <= t_wanted)
                trials += 1
                #idx = np.random.randint(0,n)
                #v[idx] = -1 * v[idx]
                
                if trials % 10000 == 0:
                    print(trials, np.exp(np.log2(trials)-n))
                    if  trials > 100000:
                        J_ij = None
                        trials = 0
                        v = None#np.random.choice([-1,1],size=n) 
                        misses += 1 
                        print(f"New_J {misses}")
            
                  
            print(h, np.exp(np.log2(trials)-n),trials)
            if h not in extras:
                del res['fields']
                del res['xfields']
            results.append(res)
            
            if len(results) % save_every == 0:
                dump_pickle(results, p)
        pbar.update(1)
    