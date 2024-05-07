import numpy as np
import matplotlib.pyplot as plt

def run_sim(n = 1000,track_h=False, final_fields=False, track_flips=False):
    
    if track_h and final_fields:
        raise ValueError('track_h and final_fields cannot be both True')

    J_ij = np.random.normal(size=(n,n)) / np.sqrt(n)
    J_ij = np.triu(J_ij, 1)
    J_ij = J_ij + J_ij.T

    config = np.random.choice([-1,1], size=n)

    def energy(config, J_ij, n):
        return - 0.5 / n  * np.sum(J_ij * np.outer(config, config)) #/ np.sqrt(n)


    positive_fields = []

    fields = (J_ij @ config).reshape(-1)

    positive_fields = (config * fields < 0).flatten()
    positive_fields_pos = np.argwhere(positive_fields)

    t = 0
    hs_min = []
    hs_max = []
    flips = np.zeros_like(config)
    energy_ = energy(config, J_ij, n)
    while len(positive_fields_pos) > 0:
        
        idx = np.abs(fields[positive_fields]).argmin()
        idx2 = np.abs(fields[positive_fields]).argmax()
        
        pos = positive_fields_pos[idx]
        
        _hs_max = fields[positive_fields][idx2]
        _hs_min = fields[positive_fields][idx]

        config[pos] *= -1
        flips[pos] += 1

        delta_field = 2 * config[pos] * J_ij[:,pos]
        fields += delta_field.reshape(-1)

        positive_fields = (config * fields < 0).flatten()
        positive_fields_pos = np.argwhere(positive_fields)
        t+=1
        energy_ += -2/n * fields[pos] * config[pos]
        
        #poss.append(positive_fields_pos.shape[0])
        if t % (100 * n) == 0:
            print(t,energy_)
            
        if track_h and t % n == 0 or len(positive_fields) == 0:
            hs_max.append(_hs_max)
            hs_min.append(_hs_min)
    

    if track_h:
        return t,n, energy(config, J_ij, n), np.abs(fields).min(), (hs_min,hs_max), fields
    elif final_fields:
        return t,n, energy(config, J_ij, n), np.abs(fields).min(), fields
    elif track_flips:
        return t,n, energy(config, J_ij, n), np.abs(fields).min(), flips
    else:
        return t,n, energy(config, J_ij, n), np.abs(fields).min()


