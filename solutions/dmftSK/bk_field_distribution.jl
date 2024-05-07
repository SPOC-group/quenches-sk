# compute the partition function at fixed x[t] * h[t]

# first for a given trajectory
function bk_xZ_at_xht(xh, t, x, params, constraints, rng)
    Q, R, V = bk_fromParamsToMatrices(params)
    T = bk_getParamsDimension(params)

    h = xh / x[t] # I am fixing xh to a certain value as I am interested to the fitness, not the field per se

    @assert length(x) == T+1
    @assert 1 <= t <= T

    xT = x[1:T]
    nott = 1:T .!= t

    boxes = x |> constraints |> compute_boxes
    integ = zero(eltype(params))

    for box in boxes
        a, b = box
        at = a[t]
        bt = b[t]

        # if the variable h[t], that we are not integrating over but still is sibject to the dynamical constraint, is not contained in its integration boundaries,it is not compatible with the constraints and the contribution of the box to the integral is zero
        if !(at <= h <= bt)
            continue
        end

        # then we do not need to perform any integral
        if T==1
            integ += gausspdf(h, (V*x[1:T])[1], 1.)
        else
            anott = a[nott]
            bnott = b[nott]
    
            if any(abs.(anott-bnott) .<= eps()) 
                continue
                # here we are skipping if any of the integration variables has domain of integration whit vanishingly small length, i.e. the integral equals zero
            end
    
            # if not skipping, split the Gaussian by conditioning in the variable h[t] into a product of two Gaussians, integrate over all other variables and multiply the result by the conditioning Gaussian, and sum this over all integration boxes
            mu_out = (V*x[1:T])[t]
    
            mu_in = (V*xT)[nott] .+ Q[nott, t] * (h - mu_out)
            sigma_in = (I+Q)[nott, nott] - Q[nott, t] * Q[nott, t]'
    
            newinteg = if length(anott) == 1
                0.5 * (
                    +erf((mu_in[1]-anott[1])/sqrt(2 * sigma_in[1,1]))
                    -erf((mu_in[1]-bnott[1])/sqrt(2 * sigma_in[1,1]))
                )
            else
                mvnormcdf(mu_in, sigma_in, anott, bnott, m=5000*T, rng=rng)[1]
            end
    
            integ += gausspdf(h, mu_out, 1.) * newinteg
        end
    end

    # finally, add the h-independent R-dependent prefactor
    prefactor = exp( 0.5 * dot(x[1:T], R, x[1:T]) )
    prefactor * integ
end

# then summing over the trajectory
function bk_Z_at_xht(xh, t, p, c, params, constraints, rng)
    trajs = filter(x -> #= x[1] == 1 && =#  x[p+1] == x[p+c+1], binaryTrajectories(p+c+1))
    res = zero(eltype(params))
    for x in trajs 
        res += bk_xZ_at_xht(xh, t, x, params, constraints, rng)
    end
    res
end

# then give the distribution by normalising with the original partition function
function bk_fields_distribution_at_xht(xh, t, p, c, dyn_params, params; epsilon = 0., normalisation = -1)
    constraints = MX_constraints(dyn_params, epsilon)
    rng = Random.default_rng()
    n = normalisation == -1 ? (2 * bk_Z(p, c, params, constraints, rng)) : normalisation
    bk_Z_at_xht(xh, t, p, c, params, constraints, rng) / n
end

##### same as above but fixing h and not xh


# first for a given trajectory
function bk_xZ_at_ht(h, t, x, params, constraints, rng)
    Q, R, V = bk_fromParamsToMatrices(params)
    T = bk_getParamsDimension(params)

    @assert length(x) == T+1
    @assert 1 <= t <= T

    xT = x[1:T]
    nott = 1:T .!= t

    boxes = x |> constraints |> compute_boxes
    integ = zero(eltype(params))

    for box in boxes
        a, b = box
        at = a[t]
        bt = b[t]

        # if the variable h[t], that we are not integrating over but still is sibject to the dynamical constraint, is not contained in its integration boundaries,it is not compatible with the constraints and the contribution of the box to the integral is zero
        if !(at <= h <= bt)
            continue
        end

        
        if T==1 # then we do not need to perform any integral. The integral equals the integrand and that's it
            integ += gausspdf(h, (V*x[1:T])[1], 1.)
        else
            anott = a[nott]
            bnott = b[nott]
    
            if any(abs.(anott-bnott) .<= eps()) 
                continue
                # here we are skipping if any of the integration variables has domain of integration whit vanishingly small length, i.e. the integral equals zero
            end
    
            # if not skipping, split the Gaussian by conditioning in the variable h[t] into a product of two Gaussians, integrate over all other variables and multiply the result by the conditioning Gaussian, and sum this over all integration boxes
            mu_out = (V*x[1:T])[t]
    
            mu_in = (V*xT)[nott] .+ Q[nott, t] * (h - mu_out)
            sigma_in = (I+Q)[nott, nott] - Q[nott, t] * Q[nott, t]'
    
            newinteg = if length(anott) == 1
                0.5 * (
                    +erf((mu_in[1]-anott[1])/sqrt(2 * sigma_in[1,1]))
                    -erf((mu_in[1]-bnott[1])/sqrt(2 * sigma_in[1,1]))
                )
            else
                mvnormcdf(mu_in, sigma_in, anott, bnott, m=5000*T, rng=rng)[1]
            end
    
            integ += gausspdf(h, mu_out, 1.) * newinteg
        end
    end

    # finally, add the h-independent R-dependent prefactor
    prefactor = exp( 0.5 * dot(x[1:T], R, x[1:T]) )
    prefactor * integ
end

# then summing over the trajectory
# notice that I needed to remove the x[1] = 1 as this quantity is not symmetric anymore! Is antisymmetric
function bk_Z_at_ht(h, t, p, c, params, constraints, rng)
    trajs = filter(x -> #= x[1] == 1 && =# x[p+1] == x[p+c+1], binaryTrajectories(p+c+1))
    res = zero(eltype(params))
    for x in trajs 
        res += bk_xZ_at_ht(h, t, x, params, constraints, rng)
    end
    res
end

# then give the distribution by normalising with the original partition function
# notice that as I am summing only over x[1] = 1 in Z by symmetry I need to divide by an additional factor 2
function bk_fields_distribution_at_ht(h, t, p, c, dyn_params, params; epsilon = 0., normalisation = -1)
    constraints = MX_constraints(dyn_params, epsilon)
    rng = Random.default_rng()
    n = normalisation == -1 ? (2 * bk_Z(p, c, params, constraints, rng)) : normalisation
    bk_Z_at_ht(h, t, p, c, params, constraints, rng) / n
end