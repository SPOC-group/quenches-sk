import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from algorithms.utils import *
import ast
import matplotlib.colors as mcolors
from pathlib import Path
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl
from scipy.optimize import curve_fit
import json

CLEAN_DIR = Path('results/clean')
CLEAN_DIR.mkdir(exist_ok=True, parents=True)
save_dir = Path('figures')


ground_state_str = 'ground state'
kappa_eo_str = '$\kappa_{GR}$'
kappa_rl_str = '$\kappa_{RL}$'
energy_str = 'energy'
cycle_len_str = 'cycle length'
time_str = 'time'
transient_str = 'convergence'
dist_to_GS_str = r'$|E_{gs}-E|$'
gs=0.763166
p_forward = '$t$'
p_backward = '$p$'

color_c1 = '#E0D84A'
color_c2 = '#4AB6E0'
color_clarge = '#5D8EA1'
color_noconv = 'red'
legend_fontsize = 9

c1_marker='v'
c2_marker='o'

single_algo_color = {
    'reluctant': '#49D1BA',
    'greedy': '#E59723',
    'random': '#D6C42A',
    'iamp':'black'}

width=5
height=3
cbar_width=0.3



show_fwd = [0,1,2,3,4,5,6,10,20,50,150,390]
colors_fwd = show_fwd
colors_bwd = [0,1,2,3,4,5,6,7,8,9,10]
cmap_bwd = plt.colormaps['Oranges']
offset=5
colors_bwd_c = [cmap_bwd((i+offset)/(len(colors_bwd)+offset)) for i in range(len(colors_bwd))][::-1]
cmap_fwd = plt.colormaps['Purples']
offset  = 3
colors_fwd_c = [cmap_fwd((i+offset)/(len(colors_fwd)+offset)) for i in range(len(colors_fwd))][::-1]

colors_fwd = {p : c for c,p in zip(colors_fwd_c,colors_fwd[::-1])}
colors_bwd = {p : c for c,p in zip(colors_bwd_c,colors_bwd[::-1])}

def plot_GS(ax,nolabel=False):
    ax.axhline(-gs,linestyle='dashed',color='grey',label=ground_state_str if not nolabel else None)
    
    
"""
    Data loading functions
"""

def load_df(exp_name,groupby=None,agg=None,pre_loaded=False):
    res = []
    results_dir = RESULT_DIR / exp_name

    print('Loading raw results')
    # load all results from files in directory into res
    for p in results_dir.glob('*.pkl'):
        try:
            res += load_pickle(p)
        except EOFError:
            print(f'Error loading {p}')

    df = pd.DataFrame(res)

    return df
    

"""
    Plotting functions
"""

def find_intersection_point(x1, y1, x2, y2, y_value):
    # Calculate the slope
    m = (y2 - y1) / (x2 - x1)
    
    # Calculate the y-intercept
    c = y1 - m * x1
    
    # Calculate the x-coordinate of the intersection point
    x_intercept = (y_value - c) / m
    
    return x_intercept, y_value

def find_intersection(m1, b1, m2, b2):
    x_intersect = (b2 - b1) / (m1 - m2)
    y_intersect = m1 * x_intersect + b1
    return x_intersect, y_intersect


# Define the parabola function
def parabola(x, a, b, c):
    return a * x**2 + b * x + c

def fit_parabola(x_data, y_data):
    popt, pcov = curve_fit(parabola, x_data, y_data)
    a, b, c = popt
    curvature = 2 * a / ((1 + b**2)**(3/2))
    return popt, curvature


def power_law_fit(x, y, a, initial_guess=(1.0, 1.0)):
    # Define the power-law function
    def func(x, b, c):
        return  b * x ** (-c) - a

    # Perform the curve fitting
    popt, _ = curve_fit(func, x, y, p0=initial_guess)

    return popt

def line_fit(x, y, initial_guess=(1.0, 1.0)):
    # Define the power-law function
    def func(x,m,b):
        return  m * x +b

    # Perform the curve fitting
    popt, _ = curve_fit(func, x, y, p0=initial_guess)

    return popt
    