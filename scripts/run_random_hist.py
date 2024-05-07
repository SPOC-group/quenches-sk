import matplotlib.pyplot as plt
import numpy as np
import algorithms.random_change as rs
from algorithms.utils import *
from tqdm import tqdm

def count_items_in_bins(values, num_bins, bin_counts, _range=(-6,6)):
    # Calculate the bin width
    vmin = _range[0]
    vmax = _range[1]
    bin_width = (vmax-vmin)/num_bins

    # Iterate through each item in the array of values
    for value in values:
        # Determine the bin index for the current value
        if value <= vmin:
            bin_index = 0  # First open bin
        elif value >= vmax:
            bin_index = num_bins - 1  # Last open bin
        else:
            bin_index = int((value - vmin) // bin_width)  # Calculate bin index

        # Increment the count for the corresponding bin
        bin_counts[bin_index] += 1


n = 100
n_samples = 100000

results_dir = RESULT_DIR / f'random_hs_last_hist'
results_dir.mkdir(exist_ok=True, parents=True)
seed = np.random.randint(0,100000)
p = results_dir / f'{seed}_hist_{n=}'

# use twdm progress bar
num_bins = 4000
bin_counts = np.zeros(num_bins)

iii = 0
while iii < n_samples:
    iii += 1
    t,n, e, h_min,fields = rs.run_sim(n,final_fields=True)
    count_items_in_bins(fields, num_bins, bin_counts)
    
    if iii % 100 == 0:
        np.save(p, bin_counts)
        print(iii/n_samples)
np.save(p, bin_counts)

