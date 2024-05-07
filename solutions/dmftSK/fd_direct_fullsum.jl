##################################
# code to solve forward dmft equations by computing fully sums and integrals
##################################

include("utils.jl")

# given Q and V up to time T-1, returns the newly computed Q(T, t) for t in 1:T-1
function fd_eqQ(x, params, constraints, rng)
    T = fd_getParamsDimension(params) + 1
    Q, V = fd_fromParamsToMatrices(params)

    result = zeros(T-1)
    ids = 1:T-1
    normx = x[ids]

    mu = (V * normx)[ids]
    sigma = (I + Q)[ids, ids]
    boxes = x |> constraints |> compute_boxes
    
    for box in boxes
        a, b = box
        if any(abs.(a-b) .<= eps())
            continue
        end
        for t in ids
            if length(a) == 1
                result[t] += x[t] * 0.5 * (
                    +erf((mu[1]-a[1])/sqrt(2))
                    -erf((mu[1]-b[1])/sqrt(2))
                )
            else
                result[t] += x[t] * mvnormcdf(mu, sigma, a, b, m=5000*(T-1), rng=rng)[1]
            end
        end
    end
    return result
end

# given Q and V up to time T-1, returns the newly computed V(T, t) for t in 1:T-1
function fd_eqV(x, params, epsilon, constraints, rng)
    T = fd_getParamsDimension(params) + 1
    Q, V = fd_fromParamsToMatrices(params)

    result = zeros(T-1)
    ids = 1:T-1
    normx = x[ids]
    boxes = x |> constraints |> compute_boxes
    for t in ids
        not_t = ids .!= t
        sigma = (I + Q)[not_t, not_t] - Q[not_t, t] * Q[not_t, t]'
        for box in boxes 
            a, b = box 
            at = a[t]
            bt = b[t]
            a_not_t = a[not_t]
            b_not_t = b[not_t]

            if any(abs.(a_not_t-b_not_t) .<= eps())
                continue
            end

            if at != -Inf
                prefactor = sqrt_nu_eps(epsilon) * gausspdf(at, (V*normx)[t], 1.)
                mu = (V * normx)[not_t] + Q[not_t, t] * (at .- (V*normx)[t])
                integral = if length(a_not_t) == 0
                    1 
                elseif length(a_not_t) == 1
                    0.5 * 
                    (
                        +erf((mu[1]-a_not_t[1])/sqrt(2))
                        -erf((mu[1]-b_not_t[1])/sqrt(2))
                    )
                else
                    mvnormcdf(mu, sigma, a_not_t, b_not_t, m=5000*(T-2), rng=rng)[1]
                end

                result[t] += prefactor * integral
            end
            
            ### lim sup
            if bt != Inf
                prefactor = -1 * sqrt_nu_eps(epsilon) * gausspdf(bt, (V*normx)[t], 1.)
                mu = (V * normx)[not_t] + Q[not_t, t] * (bt .- (V*normx)[t])
                integral = if length(b_not_t) == 0
                    1 
                elseif length(b_not_t) == 1
                    0.5 * 
                    (
                        +erf((mu[1]-a_not_t[1])/sqrt(2))
                        -erf((mu[1]-b_not_t[1])/sqrt(2))
                    )
                else
                    mvnormcdf(mu, sigma, a_not_t, b_not_t, m=5000*(T-2), rng=rng)[1]
                end

                result[t] += prefactor * integral
            end
        end
    end
    return result
end

function fd_direct_fullsum_newT(params, constraints, epsilon, rng)
    T = fd_getParamsDimension(params) + 1

    newQ = zeros(T-1)
    newV = zeros(T-1)

    # I need to sum over 
    xs = filter(x -> x[end] == 1, binaryTrajectories(T))

    for x in xs
        newQ += fd_eqQ(x, params, constraints, rng)
        newV += eta_eps(epsilon) * fd_eqV(x, params, epsilon, constraints, rng)
    end
  
    Q, V = fd_fromParamsToMatrices(params)

    finalQ = zeros(T, T)
    finalQ[1:T-1, 1:T-1] = Q
    finalQ[T, 1:T-1] = newQ
    finalQ[1:T-1, T] = newQ

    # display(Q)

    finalV = zeros(T, T)
    finalV[1:T-1, 1:T-1] = V
    finalV[T, 1:T-1] = newV

    fd_fromMatricesToParams(finalQ, finalV)
end

function fd_direct_fullsum(
    maxT, dyn_params, epsilon; 
    init = [], 
    verbose = false,
    savefile = "",
    save = false
    )

    timestamp = now()

    # set savefile name
    savefile = savefile == "" ? string(now()) : savefile 

    # set randomness
    seed = rand(UInt128)
    rng = Xoshiro(seed)

    # initialise dynamical constraint over h
    constraint = MX_constraints(dyn_params, epsilon)

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
            params = fd_direct_fullsum_newT(params, constraint, epsilon, rng)
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

                type = "fd_full_sum",
                type_settings = () # no specific setting here
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