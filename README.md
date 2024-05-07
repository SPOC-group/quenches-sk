# Code for Quenches in the Sherrington-Kirkpatrick model

This repository contains several chunks of code, necessary to reproduce the results from the above work: 
- experiments are implemented in python, folder `algorithms`
- numerical solutions to the DMFT equations are implemented in julia, folder `solutions`
- the code to plot the figures from the paper is available on the top level of this repository

## Experiments

Every algorithm discussed in the paper can be run in python via `from algorithms.<the-algo> import run_sim`.
In the directory `scripts` we provide examples for code that was used to run the experiments used in the paper. 
Results are saved in the `results/raw/` folder, and some of them may be quite heavy as each individual run is saved.
This raw data is not available in this repository, but may be reproduced from the scripts in `scripts`.
However, for most experiments we provide summary csv files in `results/clean` to facilitate secondary analyses if necessary -- some experiemnts are quite memory and compute heavy.


## Numerical solutions of DMFT equations

The implementation of the numerics takes place in the solutions folder, and is descriped in a `README.md` therein.


## Reproducing figures / accessing data 

To reproduce the figures, one can run the python notebooks on the top level of the repository.
All figures of the paper can be reproduced from the summary data in the `results/clean` directory, which is available with this repo. 
However, if experiments were run from scratch, one can uncomment the processing code and visualize the reproduced data.

## FAQ

If there are any questions, feel free to reach out to the corresponding author.

