##################################
# code to solve backtracking dmft equations by computing fully sums and integrals, and using an accelerated fixed point iterator to solve the equations
# the equations themselves are computed directly form the normalisation factor Z by forward differentiation

# in this code one can use the variable bc to set the boundary condition to x[p+1] == bc * x[p+c+1] to identify rattling attractors
##################################

function bk_xZ(x, params, constraints, rng)
    Q, R, V = bk_fromParamsToMatrices(params)
    T = bk_getParamsDimension(params)

    @assert length(x) == T+1

    mu = V*x[1:T]
    sigma = I+Q
    boxes = x |> constraints |> compute_boxes

    integ = zero(eltype(params))
    for box in boxes
        a, b = box
        if any(abs.(a-b) .<= eps())
            continue
        end
        integ += if length(a) == 1
                0.5 * (
                    +erf((mu[1]-a[1])/sqrt(2 * sigma[1,1]))
                    -erf((mu[1]-b[1])/sqrt(2 * sigma[1,1]))
                )
            else
                mvnormcdf(mu, sigma, a, b, m=5000*T, rng=rng)[1]
            end
    end

    prefactor = exp( 0.5 * dot(x[1:T], R, x[1:T]) )
    prefactor * integ
end

function bk_Z_bc(p, c, bc, params, constraints, rng)
    trajs = filter(x -> x[1] == 1 && x[p+1] == bc * x[p+c+1], binaryTrajectories(p+c+1))
    res = zero(eltype(params))
    for x in trajs 
        res += bk_xZ(x, params, constraints, rng)
    end
    res
end

function bk_gradZ_bc(p, c, bc, params, constraints, rng)
    f = let p=p, c=c, constraints=constraints, rng=rng, bc=bc
            pt -> bk_Z_bc(p, c, bc, pt, constraints, rng)
    end
    ForwardDiff.gradient(f, params)
end

function bk_updateParams_bc(p, c, bc, params, constraints, epsilon, rng)
    ne = eta_eps(epsilon)
    gradients = bk_gradZ_bc(p, c, bc, params, constraints, rng)
    normalisation = bk_Z_bc(p, c, bc, params, constraints, rng) 
    gradQ, gradR, gradV = bk_fromParamsToMatrices(gradients)
    updateQ = gradR
    updateR = gradQ
    updateV = gradV' * ne
    bk_fromMatricesToParams(updateQ, updateR, updateV) / normalisation
end

function bk_entropy_bc(p, c, bc, params, constraints, epsilon, rng)
    ne = eta_eps(epsilon)
    Q, R, V = bk_fromParamsToMatrices(params)
    s = log(2)
    s -= 0.5 * tr(R * Q) # R and Q have zero diagonal and are symmetric
    s -= (ne == 0 ? 0 : tr(V*V) / (2*ne))
    s += log(bk_Z_bc(p, c, bc, params, constraints, rng))
    s
end

function bk_solve_bc(p::Int, c::Int, bc::Int, dyn_params, epsilon; 
    init = [],
    iters = 500,
    beta = 1.0,
    tol = 1e-3,
    m=3,
    verbose = false,
    seed = -1
    ) 

    timestamp = now()

    # set randomness
    seed = ifelse(seed == -1, rand(UInt128), seed)
    rng = Xoshiro(seed)

    T = p+c
    params = init == [] ? zeros(2*T^2 - T) : init
    energy = bk_energy(params, epsilon = epsilon)

    constraints = MX_constraints(dyn_params, epsilon)

    function updater!(F, x)
        F[:] = bk_updateParams_bc(p, c, bc, x, constraints, epsilon, rng)
    end

    res = fixedpoint(updater!, params, iterations = iters, ftol=tol, m=m, beta=beta, show_trace=verbose)
    energy = !converged(res) ? NaN : bk_energy(res.zero, epsilon=epsilon)
    norm_entropy = !converged(res) ? NaN : bk_entropy_bc(p, c, bc, res.zero, constraints, epsilon, rng)/log(2)

    data = (
        p = p,
        c = c,
        bc = bc,

        keo = dyn_params[1],
        krl = dyn_params[2],
        epsilon = epsilon,

        params = res.zero,
        energy = energy,

        norm_entropy = norm_entropy,

        fp_tol = res.residual_norm,    # ftol at end of iteration
        fp_iters_used = res.iterations, # iters used at end of iteration 

        seed = seed,
        timestamp = timestamp,

        type = "bk_iteration",
        type_settings = (
            iters=iters,  # number of fp iterations allowed
            beta=beta,   # beta parameter for NLSolve, analogous of learning rate
            m=m,      # m parameter for NLSolve, number of previous iterates used in acceleration
            tol=tol,   # tolerance on the residual f(x)-x in the fp iteration
            init=init,   # initialisation of iteration
        )
    )   

    return data
end

function bk_sweep_kappa_bc(
    p, c, bc, dyn_params, epsilon; 
    init = [],
    iters = 500,
    tol = 1e-3,
    verbose = false,
    fp_beta = 1.,
    fp_m = 5,
    savefile = "",
    save = false,
    )

    # set savefile name
    savefile = savefile == "" ? string(now()) : savefile 

    init = init
    data = []
    res = []
    
    for dp in dyn_params 
        if verbose
            println()
            println("Parameters: ", dp)
        end

        try
            res = bk_solve_bc( 
                p, c, bc, dp, epsilon, 
                init = [],
                iters = iters,
                beta = fp_beta,
                tol = tol,
                m = fp_m,
                verbose = verbose,
                seed = -1
                ) 
            init = res.fp_iters_used != res.type_settings[1] ? res.params : []
        catch e
            println(e)
        end

        data = if save 
            try 
                f = jldopen(filename)
                d = f["data"]
                close(f)
                d
            catch e
                println(e)
                []
            end
        else
            data
        end

        push!(data, res)

        if save
            jldsave(savefile; data)        
        end
    end

    return data
end