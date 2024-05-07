##################################
# code to solve forward dmft equations by sampling the stochastic process, without reusing previous samples
##################################

include("utils.jl")

# generates new path
function fd_sample_path(M, params, dynamics, epsilon; rng = Random.default_rng())
   
    T = fd_getParamsDimension(params) #current dimension of paramters, we will estimate Q(t, T+1) and V(T+1, t) for t=1, ..., T
    snue = sqrt_nu_eps(epsilon)

    Q, V = fd_fromParamsToMatrices(params)
    L = (cholesky(I+Q).U)' # such that L L' = I+Q, and phi = L * z has covariance I+Q if z has covariance I

    # preallocate
    x = ones(Int, (T+1, M))
    phi = L * randn(rng, (T, M))

    for _ in 1:T+1
        for t in 2:T+1
            x[t, :] = dynamics.( 
                    x[t-1, :],
                    snue * (
                        phi[t-1, :] 
                        .+ V[t-1:t-1, 1:t-1] * x[1:t-1, M]
                        ), 
                )
        end
    end

    x, phi
end

# takes parameters Q, V and computes Q, V at the next time step by sampling ex-novo the stochastic process 
function fd_direct_sampling_newT(params, dynamics, epsilon, rng, max_step)
   
    T = fd_getParamsDimension(params) #current dimension of paramters, we will estimate Q(t, T+1) and V(T+1, t) for t=1, ..., T
    snue = sqrt_nu_eps(epsilon)

    Q, V = fd_fromParamsToMatrices(params)
    L = (cholesky(I+Q).U)' # such that L L' = I+Q, and phi = L * z has covariance I+Q if z has covariance I

    # preallocate
    x = ones(Int, T+1)
    phi = zeros(T)
    newxx = zeros(T)
    newxphi = zeros(T)

    for _ in 1:max_step
        phi = L * randn(rng, T)

        x[1] = 1
        for t in 2:T+1
            x[t] = dynamics( 
                    x[t-1],
                    snue * (phi[t-1] + dot(V[t-1, 1:t-1],x[1:t-1])), 
                )
        end

        newxx =  newxx + x[T+1] * x[1:T]
        newxphi =  newxphi + x[T+1] * phi 
    end

    newxx /= max_step
    newxphi = newxphi /= max_step

    newQ = newxx # T dimensional
    newV = snue^2 * inv(I + Q) * newxphi # T dimensional

    ## update Q
    finalQ = zeros((T+1, T+1)) 
    finalQ[1:T, 1:T] = Q
    finalQ[T+1, 1:T] = newQ
    finalQ[1:T, T+1] = newQ
    Q = finalQ 

    # display(Q)

    ## update V
    finalV = zeros((T+1, T+1))
    finalV[1:T, 1:T] = V
    finalV[T+1, 1:T] = newV
    V = finalV

    fd_fromMatricesToParams(Q, V)
end

# starting from init (and its time dimensionality startT), extend the parameters init = (Q,V) up to time maxT
# if it fails, tipically due to failing cholesky decomposition of I+Q, it stops and returns the last timestep computed
function fd_direct_sampling(
    maxT, dyn_params, epsilon; 
    init = [], 
    verbose = false,
    num_paths = 1000000, 
    savefile = "",
    save = false
    )

    timestamp = now()

    # set savefile name
    savefile = savefile == "" ? string(now()) : savefile 

    # set randomness
    seed = rand(UInt128)
    rng = Xoshiro(seed)

    # initialise dynamics sigma(x,h)
    dynamics = MX_dynamics(dyn_params)

    # initialise parameters if needed
    params = init == [] ? Vector{Float64}() : init
    startT = fd_getParamsDimension(params)
    
    @assert startT <= maxT
    
    data = ()

    for t in startT:maxT
        if verbose
            println("Computing t = ", fd_getParamsDimension(params)+1)
        end

        try
            params = fd_direct_sampling_newT(params, dynamics, epsilon, rng, num_paths)
            energy = fd_energy(params, epsilon=epsilon)
            data = (
                p = t,
                c = 0,

                keo = dyn_params[1],
                krl = dyn_params[2],
                epsilon = epsilon,

                params = params,
                energy = energy,

                seed = seed,
                timestamp = timestamp,

                type = "fd_direct_sampling",
                type_settings = (num_paths=num_paths) # number of paths simulated
            )   
        catch e
            println(e)
        end
        
        if save
            jldsave(savefile; data)
        end
    end
    return data
end