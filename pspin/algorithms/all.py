import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def generate_J_ijk(n):
    J_ijk = np.zeros((n,n,n))
    #np.random.seed(seed)
    factor = np.sqrt(3/(n**2))
    # symmetrices the matrix, puts zero on the diagonal
    for i in range(n): # <=
        for j in range(i): # < 
            for k in range(j): # < 
                value = np.random.normal() * factor
                
                # assign all permutations with the value
                J_ijk[i,j,k] = value
                J_ijk[i,k,j] = value
                J_ijk[k,i,j] = value
                J_ijk[k,j,i] = value
                J_ijk[j,k,i] = value
                J_ijk[j,i,k] = value
        J_ijk[i,i,i] = 0
    return J_ijk

@njit
def get_energy(sigma, J_ijk, n):
    E = 0
    for i in range(n):
        for j in range(i):
            for k in range(j):
                E += J_ijk[i,j,k] * sigma[i] * sigma[j] * sigma[k]
    return -1/n * E # n**2??


@njit
def get_fields(sigma, J_ijk, n):
    fields = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(j):
                if k == i:
                    continue
                # add the change
                fields[i] += J_ijk[i,j,k] * sigma[j] * sigma[k]
    return fields

@njit
def update_fields(i,fields, sigma, J_ijk, n):
    for j in range(n):
        for k in range(n):
            if k == i:
                continue
            fields[j] += 2 * J_ijk[i,j,k] * sigma[i] * sigma[k]
    return fields

def run_greedy(n,t_max=3000,record_energy=False):
    J_ijk = generate_J_ijk(n)
    config = np.random.choice([-1,1], size=n)
    E = get_energy(config, J_ijk, n)
    fields = get_fields(config, J_ijk, n)
    positive_fields = (config * fields < 0).flatten()
    positive_fields_pos = np.argwhere(positive_fields)[:,0]

    t = 0
    E = get_energy(config, J_ijk, n)
    Es = []

    while len(positive_fields_pos) > 0 and t < n * t_max:

            idx = np.abs(fields[positive_fields]).argmax()
            pos = positive_fields_pos[idx]

            config[pos] *= -1

            update_fields(pos,fields, config, J_ijk, n)

            positive_fields = (config * fields < 0).flatten()
            positive_fields_pos = np.argwhere(positive_fields)[:,0]


            t+=1
            E += -2/n * fields[pos] * config[pos]
            if record_energy:
                Es.append(E)
    E = get_energy(config, J_ijk, n)
    
    if t == t_max:
        E = np.nan
    return n, E, Es, t

def run_reluctant(n,record_energy=False):
    J_ijk = generate_J_ijk(n)
    config = np.random.choice([-1,1], size=n)
    E = get_energy(config, J_ijk, n)
    fields = get_fields(config, J_ijk, n)
    positive_fields = (config * fields < 0).flatten()
    
    positive_fields_pos = np.argwhere(positive_fields)[:,0]
    t = 0
    E = get_energy(config, J_ijk, n)
    Es = []
    while len(positive_fields_pos) > 0:

            idx = np.abs(fields[positive_fields]).argmin()
            pos = positive_fields_pos[idx]

            config[pos] *= -1
            update_fields(pos,fields, config, J_ijk, n)
            
            #other_fields = get_fields(config, J_ijk, n)
            #assert np.allclose(fields, other_fields), "no"

            positive_fields = (config * fields < 0).flatten()
            positive_fields_pos = np.argwhere(positive_fields)[:,0]

            t += 1
            E += -2/n * fields[pos] * config[pos]
            #E_true = get_energy(config, J_ijk, n)
            #assert np.isclose(E,E_true), f"{E}, {E_true}"
            if record_energy:
                Es.append(E)
    E = get_energy(config, J_ijk, n)
    
    return n, E, Es, t