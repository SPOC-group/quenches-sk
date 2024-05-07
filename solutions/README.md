# Code to solve the DMFT equations

This folder contains the code to solve the DMFT equations, as well as the code to parse the raw data into the plotted data of the article.
The code is written in Julia, and tested in Julia 1.8 and 1.10. 
We provide a `Project.toml` file specifying the dependencies.

- `dmftSK` contains the DMFT equations solver. It can be imported using `include("dmftSK/main.jl")`
- `data_*_raw.jld2` contains the raw solution of the DMFT equations for various values of the parameters, for both forward `fd` and backtracking `bk` DMFT.
- `generate_csv_figures.jl` processes the raw data into the data used to plot the figures of the article.

## Forward DMFT

To solve the forward DMFT equations with the "direct sampling" method (see SM, Supplementary Material of the article), you can use the function `fd_direct_sampling` defined in the file `dmftSK/fd_direct_sampling.jl`. We also provide solvers for the other methods mentioned in the SM in the files `fd_direct_sampling_reuse.jl` (direct sampling with reuse solution) and `fd_direct_fullsum.jl` (direct exact solution).

```
function fd_direct_sampling(
    maxT,                   # time up to which the equations are solved
    dyn_params,             # parameters defining the dynamical rule: [kgr, krl]
    epsilon;                # parameter defining the asymmetry of the couplings J. epsilon = 0 for symmetric J
    init = [],              # initial parameters (upper triangle of Q, lower triangle of V, 
                            # flattened and concatenated). If [], starts at time t=1 with the analytical solution
    verbose = false,        # if true, print progress informations to stdout
    num_paths = 1000000,    # number of samples used to approximate Monte Carlo algorithms
    save = false,           # if true, save results to savefile
    savefile = "",          # file to which the results should be saved
    )
```

See the `dmftSK/utils.jl` file for details on how the order parameters are stored in a flattened way.

`data_fd_raw.jld2` contains the solution of the forward DMFT equations for various values of `kgr` and `krl`. 
We provide only solutions for the largest time we looked at, as all previous times solutions can be recovered by considering the appropriate subset of order parameters, restricted to the times of interest.

On normal CPUs, it is possible to solve the equations quite fast, in the order of some seconds per time step, up to `maxT ~ 400`. Speed-ups could be achieved by simple parallelisation of the Monte Carlo integrals.

## Backtracking DMFT

To solve the backtracking DMFT equations, you can use the functions `bk_solve` and `bk_sweep_kappa` defined in the file `dmftSK/bk_iteration.jl`.    

`bk_solve` solves the DMFT equations for a given value of the dynamical parameters `kgr` and `krl`.
```
bk_solve(
    p,                      # length of the transient
    c,                      # length of the cycle
    dyn_params,             # parameters defining the dynamical rule: [kgr, krl]
    epsilon;                # parameter defining the asymmetry of the couplings J. epsilon = 0 for symmetric J
    init = [],              # initial parameters. If [], initialises all correlators to 0
    iters = 500,            # max number of iterations for the equation solver
    beta = 1.0,             # learning rate of the fixed point iterator, see NLSolve.jl
    m=3,                    # amount of previous iterates considered in the fixed point iterator, see NLSolve.jl
    tol = 1e-3,             # tolerance at which the equations are considered solved
    verbose = false,        # if true, print progress informations to stdout
    seed = -1               # seed for the rng. If -1, a random seed is used
) 
```

`bk_sweep_kappa` solves the DMFT equations for a set of values of the dynamical parameters `kgr` and `krl`, using the solution at a given value as initial condition for the solver for the next value of the dynamical parameters. This allows to speed up the solver by leveraging approximate solutions whenever paramters change only slightly.

```
bk_sweep_kappa(
    p,                  # length of the transient
    c,                  # length of the cycle
    dyn_params,         # vector of parameters defining the dynamical rule: [[kgr_1, krl_1], [kgr_2, krl_2], ...]
    epsilon;            # parameter defining the asymmetry of the couplings J. epsilon = 0 for symmetric J
    init = [],          # initial parameters. If [], initialises all correlators to 0
    iters = 500,        # max number of iterations for the equation solver
    tol = 1e-3,         # tolerance at which the equations are considered solved
    verbose = false,    # if true, print progress informations to stdout
    fp_beta = 1.,       # learning rate of the fixed point iterator, see NLSolve.jl
    fp_m = 5,           # amount of previous iterates considered in the fixed point iterator, see NLSolve.jl
    save = false,       # if true, save results to savefile
    savefile = "",      # file to which the results should be saved
)
```

On normal CPUs, it is possible to solve the equations quite fast for `p+c <= 5`. Larger values of `p+c` slow down exponentially fast.

The file `dmftSK/bk_field_distribution.jl` provides utilities to compute the field distribution and the fitness distribution, while `dmftSK/bk_iteration_rattle.jl` provides a solver for the backtracking DMFT equations with rattling cycles, i.e. `x[p+1] = bc * x[p+c+1]` for `bc = +1` (non-rattling) or `bc = -1` (rattling).
