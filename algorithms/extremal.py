import numpy as np
import matplotlib.pyplot as plt

def run_sim(n = 1000,track_h=False, final_fields=False, track_flips=False):

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
    hs_min = []
    hs_max = []
    t = 0
    energy_ = energy(config, J_ij, n)
    flips = np.zeros_like(config)
    while len(positive_fields_pos) > 0:
        idx = np.abs(fields[positive_fields]).argmax()
        idx2 = np.abs(fields[positive_fields]).argmin()
        if track_h:
            hs_min.append(fields[positive_fields][idx2])
            hs_max.append(fields[positive_fields][idx])
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
        if t % (100 * n) == 0:
            print(t,energy_)

    if track_h:
        return t,n, energy(config, J_ij, n), np.abs(fields).min(), (hs_min, hs_max), fields
    elif final_fields:
        return t,n, energy(config, J_ij, n), np.abs(fields).min(), fields
    elif track_flips:
        return t,n, energy(config, J_ij, n), np.abs(fields).min(), flips
    else:
        return t,n, energy(config, J_ij, n), np.abs(fields).min()


