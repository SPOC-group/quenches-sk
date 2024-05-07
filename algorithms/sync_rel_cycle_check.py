from numba import njit
import numpy as np

# compiled version with inplace operations

@njit
def update(from_config,to_config,local_field,n,h):
    changes = 0
    for i in range(n):
        if local_field[i] > 0 and local_field[i] < h*np.sqrt(n):
            to_config[i] = +1
            changes += 1
        elif local_field[i] < 0 and local_field[i] > -h*np.sqrt(n):
            to_config[i] = -1
            changes += 1
        else:
            to_config[i] = from_config[i]
    return changes

@njit
def triangulate(J_ij,n):
    # symmetrices the matrix, puts zero on the diagonal
    for i in range(n):
        for j in range(i):
            J_ij[i][j] = J_ij[j][i]
        J_ij[i][i] = 0
            
@njit
def all_equal(A,B,n):
    for i in range(n):
        if A[i] != B[i]:
            return False
    return True

@njit
def energy(x,J_ij,n):
    energy = 0
    for i in range(n):
        for j in range(n):
            energy += x[i]*x[j]*J_ij[i,j]
    return -0.5/(n*np.sqrt(n)) * energy

def run_sim(n, h, record_energy=False,seed_J=None,seed_x=None,max_t=np.inf):
    
    if seed_J is None:
        seed_J = np.random.randint(0,100000)
    if seed_x is None:
        seed_x = np.random.randint(0,100000)
    
    J_ij = np.random.RandomState(seed=seed_J).normal(size=(n,n))
    triangulate(J_ij,n)
    
    density = int(n/2)
    v = np.random.RandomState(seed=seed_x).permutation(np.concatenate((np.ones(density), -1 * np.ones(n - density)))) 
    
    
    c1= v
    c2=np.empty_like(v)
    
    t = 0
    elocals = [energy(c1,J_ij,n)]
    configs = [np.copy(c1)]
    hs = []
    
    save_last_7_x = [None] * 7
    save_last_7_h = [None] * 7
    
    
    while t < max_t:
        local_field = (c1 @ J_ij)
        hs.append(local_field)
        save_last_7_x[t%7] = np.copy(c1)
        save_last_7_h[t%7] = np.copy(local_field)
        changes = update(from_config=c1,to_config=c2,local_field=local_field,n=n,h=h)  
        temp = c2
        c2 = c1
        c1 = temp
        t+=1
        if record_energy:
            elocals.append(energy(c1,J_ij,n))
            configs.append(np.copy(c1))
    save_last_7_x[(t+1)%7] = np.copy(c1)
    save_last_7_h[(t+1)%7] = np.copy(local_field)       
    hs.append(c1 @ J_ij)
            
    for i in reversed(range(len(configs)-1)):
        if all_equal(configs[-1],configs[i],n):
            break
        else:
            i = -1
            
    cycle_length = len(configs) - i - 1 if i != -1 else None
    print('ok',cycle_length)
    if cycle_length is not None:
        while i >= 0:
            i -= cycle_length
            #print(f'checking step {i}')
            if not all_equal(configs[-1],configs[i],n):
                i += cycle_length
                break
        # now we found the first time that the last configuration was seen
        # we now have to find the true transient length, i.e. where we enter into the periodicity
        #print(i)
        for k in range(cycle_length):
            if i-k < 0:
                break
            if not all_equal(configs[-1-k],configs[i-k],n):
                break
            
        t = i - k # length 0 -> cycle starts immediately
        #print(t,elocals[t])
            
        
    return {
            'energy_trajectory': elocals, 
            'energy': energy(c1,J_ij,n), 
            'transient': t, 
            'c':  cycle_length,
            'last_7': (configs[max(t-4,0):min(t+4,max_t)],hs[max(t-4,0):min(t+4,max_t)]) if cycle_length==4 else None,
            'seed_J':seed_J, 
            'seed_x':seed_x,
            'n':n,
            'h':h,
            }