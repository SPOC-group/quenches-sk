##################################
# code to solve forward dmft equations by sampling the stochastic process, reusing previous samples
##################################

function fd_dsr_initialise(M, rng; type = Float64)
    randn(rng, type, 1, M), ones(1, M), zeros(type, 0)
end

## ATTENTION
# next function is supposed to fail around convergence, due to the fact that Q(t, t+1) = 1 usually
# we do not treat this case explicitly, so this function should always be wrapped into a try...catch statement

# x is a matrix of size (T, M)
# phi is a matrix of size (T, M)
# Q and V are matrices of size T
# the new step we compute is T+1
# dynamics is the function x' = sigma(h, x)

# I tried to preallocate x and phi but it does not seem to help
function fd_dsr_newT(phi, x, params, dynamics, epsilon, rng)

    T, M = size(x)
    @assert size(x) == size(phi)
    @assert T == fd_getParamsDimension(params)

    ne = eta_eps(epsilon) 
    snue = sqrt_nu_eps(epsilon)
    Q, V = fd_fromParamsToMatrices(params)

    ## compute new_x of shape M
    new_x = dynamics.( 
                x[T, :],
                snue * (phi[T, :] + x' * V[T, :])
            )
    
    ## compute new elements of Q (T of them), i.e. <x_s x_{T+1}> for 1 <= s <= T
    newQ = (x * new_x) / M
    
    ## update x 
    x = vcat(x, new_x')

    ## update Q
    finalQ = zeros(eltype(Q), (T+1, T+1)) # finalQ = zeros((T+1, T+1))
    finalQ[1:T, 1:T] = Q
    finalQ[T+1, 1:T] = newQ
    finalQ[1:T, T+1] = newQ
    Q = finalQ 

    ## need to sample M-times phi(T+1), which is Gaussian together with the previous phis
    invcov = inv(I+Q[1:T, 1:T])
    crosscov = Q[1:T, T+1]

    ## this is a (1, M) vector of scalar means for all the trajectories
    mu = phi' * invcov * crosscov
    ## the covariance is the same for all the trajectories
    sigma = (1 .- crosscov' * invcov * crosscov)[1] |> abs |> sqrt
    
    new_phi = mu + sigma * randn(rng, eltype(phi), M) 

    ## update phi, is size (T+1, M)
    phi = vcat(phi, new_phi')

    ## compute <x_{T+1} phi_s> of shape T+1, for 1 <= s <= T+1
    x_phi = phi * new_x / M

    ## compute new elements of V (T+1, but the last one should be zero)
    newV = ne * invcov * x_phi[1:T]
    # discard last element 
    # newV = newV[1:T]

    ## update V
    finalV = zeros(eltype(V),(T+1, T+1)) # finalV = zeros(( T+1, T+1))
    finalV[1:T, 1:T] = V
    finalV[T+1, 1:T] = newV
    V = finalV

    (phi, x, fd_fromMatricesToParams(Q, V))
end

function fd_direct_sampling_reuse(
    maxT, dyn_params, epsilon; 
    num_paths = 1000000,
    verbose = false,
    savefile = "",
    save = false,
    saverate = 10
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
    phi, x, params = fd_dsr_initialise(convert(Int,num_paths), rng)
    
    data = ()

    for t in 1:maxT
        if verbose
            println("Computing t = ", fd_getParamsDimension(params)+1)
        end

        try
            phi, x, params = fd_dsr_newT(phi, x, params, dynamics, epsilon, rng)
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

                type = "fd_direct_sampling_reuse",
                type_settings = (num_paths=num_paths) # number of paths simulated
            )   
        catch e
            println(e)
        end
        
        if save && (t % saverate == 0)
            jldsave(savefile; data)
        end
    end
    return data
end