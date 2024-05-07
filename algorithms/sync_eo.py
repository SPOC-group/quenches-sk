from numba import njit
import numpy as np

# compiled version with inplace operations
"""
@njit
def update(from_config,to_config,local_field,n,h):
    changes = 0
    for i in range(n):
        if local_field[i] > h*np.sqrt(n):
            to_config[i] = +1
            changes += 1
        elif local_field[i] < -h*np.sqrt(n):
            to_config[i] = -1
            changes += 1
        else:
            to_config[i] = from_config[i]
    return changes
"""

@njit
def update(from_config,to_config,local_field,n,h):
    changes = 0
    for i in range(n):
        if -1 * from_config[i] * local_field[i] > h*np.sqrt(n):
            to_config[i] = -1 * from_config[i]
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

def run_sim(n, h, record_energy=False,seed_J=None,seed_x=None,max_t=np.inf,record_fields=False,J_ij=None,v=None):
    
    if seed_J is None:
        # check if seed_J is an integer
        if not isinstance(seed_J, int) and J_ij is not None:
            # using given J_ij
            pass
        else:
            seed_J = np.random.randint(0,100000)
            J_ij = np.random.RandomState(seed=seed_J).normal(size=(n,n))
            triangulate(J_ij,n)
    if seed_x is None:
        seed_x = np.random.randint(0,100000)
    

    density = int(n/2)
    if v is None:
        v = np.random.RandomState(seed=seed_x).permutation(np.concatenate((np.ones(density), -1 * np.ones(n - density))))
        #v = np.random.RandomState(seed=seed_x).choice([-1,1],size=n) 
    else:
        pass
    
    c1= v
    c2=np.empty_like(v)
    c3=np.empty_like(v)
    
    c1_is_c3 = False
    c1_is_c2 = False
    t = 0
    elocals = [energy(c1,J_ij,n)]
    
    while  (not c1_is_c3 and not c1_is_c2) and t < max_t:
        local_field = (c1 @ J_ij)
        changes = update(from_config=c1,to_config=c3,local_field=local_field,n=n,h=h)  
        temp = c2
        c2 = c1
        c1 = c3
        c3 = temp
        
        c1_is_c3 = all_equal(c1,c3,n)
        c1_is_c2 = all_equal(c1,c2,n)
        t+=1
        if record_energy:
            elocals.append(energy(c1,J_ij,n))
        
    if c1_is_c2:
        result = {
            'energy_trajectory': elocals, 
            'energy': energy(c1,J_ij,n), 
            'transient': t,
            'c': 1, 
            'seed_J':seed_J, 
            'seed_x': seed_x,
            'n':n,
            'h':h,
            }
        
    elif c1_is_c3:
        result =  {
            'energy_trajectory': elocals, 
            'energy': 0.5 * (energy(c2,J_ij,n)+ energy(c1,J_ij,n)), 
            'e1': energy(c2,J_ij,n),
            'e2': energy(c1,J_ij,n),
            'transient': t, 
            'c':2, 
            'seed_J':seed_J, 
            'seed_x':seed_x,
            'n':n,
            'h':h,
            }
        
    else:
        result ={
            'energy_trajectory': elocals, 
            'energy': 0.5 * (energy(c2,J_ij,n)+ energy(c1,J_ij,n)), 
            'e1': energy(c2,J_ij,n),
            'e2': energy(c1,J_ij,n),
            'transient': t, 
            'c': None,
            'seed_J':seed_J, 
            'seed_x':seed_x,
            'n':n,
            'h':h,
            }
    if record_fields:
        result['fields'] = c1 @ J_ij
        result['xfields'] = c1 * (c1 @ J_ij)
    
    return result, J_ij