import numpy as np
import matplotlib.pyplot as plt

def run_sim(n = 1000, track_h=False, final_fields=False, track_flips=False):

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
    energy_ = energy(config, J_ij, n)
    max_time = n * n
    flips = np.zeros_like(config)
    hs_min = []
    hs_max = []
    while len(positive_fields_pos) > 0 and t < max_time:
        idx_size = len(fields[positive_fields])
        idx = np.random.randint(0, idx_size)
        idx_min = np.abs(fields[positive_fields]).argmin()
        idx_max = np.abs(fields[positive_fields]).argmax()
        
        if track_h:
            hs_max.append(fields[positive_fields][idx_max])
            hs_min.append(fields[positive_fields][idx_min])
        pos = positive_fields_pos[idx]

        config[pos] *= -1
        flips[pos] += 1

        delta_field = 2 * config[pos] * J_ij[:,pos]
        fields += delta_field.reshape(-1)

        positive_fields = (config * fields < 0).flatten()
        positive_fields_pos = np.argwhere(positive_fields)
        t+=1
        energy_ += -2/n * fields[pos] * config[pos]
        
        #poss.append(positive_fields_pos.shape[0])
        #if t % (10) == 0:
        #    print(t,energy_)
    if t == max_time:
        print('OUT OF TIME')

    if track_h:
        return t,n, energy(config, J_ij, n), np.abs(fields).min(), (hs_min,hs_max), fields
    elif final_fields:
        return t,n, energy(config, J_ij, n), np.abs(fields).min(), fields
    elif track_flips:
        return t,n, energy(config, J_ij, n), np.abs(fields).min(), flips
    else:
        return t,n, energy(config, J_ij, n), np.abs(fields).min()


