##################################
# packages
##################################

using Dates
using Random
using LinearAlgebra
using SpecialFunctions
using MvNormalCDF
using ForwardDiff
using NLsolve
using JLD2
using DataFrames
using CSV
using Statistics

##################################
# constants
##################################

sqrt_nu_eps(epsilon) = sqrt((1-epsilon)^2 + epsilon^2)
kappa_eps(kappa, epsilon) = kappa / sqrt_nu_eps(epsilon)
eta_eps(epsilon) = ((1-epsilon)^2 - epsilon^2) / ((1-epsilon)^2 + epsilon^2)

##################################
# helper functions
##################################

# all +-1 strings of length T
function binaryTrajectories(T)
    n = T
    traj = []
    for i = 0:2^n-1
        s = bitstring(i)
        s = s[length(s)-n+1:end]
        s = parse.(Int, collect(s))
        s = 2*s .- 1
        push!(traj, s)
    end
    traj
end

# compute pairs of T objects
getPairs(T) = div(T^2-T, 2)

# pdf of gaussian
gausspdf(x, mu, sigma) = exp(-0.5 * dot(x-mu, inv(sigma), x-mu)) / sqrt((2pi)^length(x) * det(sigma))

##################################
# reshaping 
##################################

# function that takes a vector [Q12, Q13, ..., Q1n, Q23, ..., Q2n, ..., Q(n-1)n]
# and construct the matrix
# [[ 0      Q12     Q13     ...     Q1n ]
#  [ Q12    0       Q23     ...     Q2n ]
#    ...    ...     ...     ...     ...
#  [ Q1n    Q2n     Q3n     ...     0   ]]
function vecToSym(vec)
    n = convert(Int, (1 + sqrt(1 + 8 * length(vec)))/2) 
    mat = zeros(eltype(vec), (n, n))
    k = 1
    for i in 1:n
        for j in (i+1):n
            mat[i, j] = vec[k]
            mat[j, i] = vec[k]
            k+=1
        end
    end
    mat
end

# inverse of vecToSym: takes the upper triangle (diagonal excluded) of a square matrix mat and flattens it into a vector
function symToVec(mat)
    @assert size(mat)[1] == size(mat)[2]
    n = size(mat)[1]
    vec = Vector{eltype(mat)}()
    for i in 1:n
        for j in (i+1):n
            push!(vec, mat[i,j])
        end
    end
    vec
end

# flatten matrix into vector row by row
matToVec(mat) = reduce(vcat, mat')

# inverse of myflat, valid for flattened square matrices
function vecToMat(vec)
    n = convert(Int, sqrt(length(vec))) 
    mat = zeros(eltype(vec), (n, n))
    k = 1
    for i in 1:n
        for j in 1:n
            mat[i, j] = vec[k]
            k+=1
        end
    end
    mat
end

# function that takes a vector [Q12, Q13, ..., Q1n, Q23, ..., Q2n, ..., Q(n-1)n]
# and construct the matrix
# [[ 0      0       0       ...     0 ]
#  [ Q12    0       0       ...     0 ]
#    ...    ...     ...     ...     ...
#  [ Q1n    Q2n     Q3n     ...     0   ]]
function vecToTri(vec)
    n = convert(Int, (1 + sqrt(1 + 8 * length(vec)))/2) 
    mat = zeros(eltype(vec), (n, n))
    k = 1
    for i in 1:n
        for j in (i+1):n
            # mat[i, j] = vec[k]
            mat[j, i] = vec[k]
            k+=1
        end
    end
    mat
    # mat |> LinearAlgebra.LowerTriangular
end

# inverse of vecToTri: takes the lower triangle (diagonal excluded) of a square matrix mat and flattens it into a vector
function triToVec(mat)
    @assert size(mat)[1] == size(mat)[2]
    n = size(mat)[1]
    vec = Vector{eltype(mat)}()
    for i in 1:n
        for j in (i+1):n
            push!(vec, mat[j,i])
        end
    end
    vec
end

##################################
# forward dmft
##################################

fd_getParamsDimension(params) = convert(Int, (1 + sqrt(1 + 4 * length(params)))/2)

# in forward DMFT params is the concatenation of entries of a symmetrix TxT matrix Q, 
# and a lower triangular/zero diagonal TxT matrix V
# the total length of the concatenation must be equal to T^2 - T
function fd_fromParamsToMatrices(params)
    if params == []
        zeros(eltype(params), (1,1)), zeros(eltype(params), (1,1))
    else
        l = getPairs(fd_getParamsDimension(params))
        vecToSym(params[1:l]), vecToTri(params[l+1:2l])
    end
end

# inverse of function above
fd_fromMatricesToParams(Q, V) = vcat(symToVec(Q), triToVec(V))

function fd_energy(params; epsilon = 0.)
    ne = eta_eps(epsilon)
    sqrt_nu = sqrt((1+epsilon)^2 + epsilon^2)
    Q, V = fd_fromParamsToMatrices(params)
    (1 + ne)/(2ne) * sqrt_nu * diag(Q*V')
end

function fd_downsizeParams(newT, params)
    @assert 1 <= newT <= fd_getParamsDimension(params)
    Q, V = fd_fromParamsToMatrices(params)
    Q = Q[1:newT, 1:newT]
    V = V[1:newT, 1:newT]
    fd_fromMatricesToParams(Q, V)
end

# function fd_localFieldDistribution(t, params)
#     Q, V = fd_fromParamsToMatrices(fd_downsizeParams(t, params))
#     trajs = binaryTrajectories(t)
#     let Q=Q, V=V
#         h -> 
#     end
# end

##################################
# backtrack dmft
##################################

bk_getParamsDimension(params) = convert(Int, (1 + sqrt(1 + 8 * length(params)))/4)

# params is a vector concatenating the upper triangle of Q, the one of R, and the rows of S
# this function extracts Q, R and V and builds the correspondin matrices
function bk_fromParamsToMatrices(params)
    if params == []
        zeros(eltype(params), (1,1)), zeros(eltype(params), (1,1)), zeros(eltype(params), (1,1))
    else
        l = getPairs(bk_getParamsDimension(params))
        vecToSym(params[1:l]), vecToSym(params[l+1:2l]), vecToMat(params[2l+1:end])
    end
end

# this is the inverse of the previous function
bk_fromMatricesToParams(Q, R, V) = vcat( symToVec(Q), symToVec(R), matToVec(V) )

function bk_energy(params; epsilon = 0.)
    ne = eta_eps(epsilon)
    sqrt_nu = sqrt_nu_eps(epsilon)
    Q, _, V = bk_fromParamsToMatrices(params)
    (1 + ne)/(2ne) * sqrt_nu * diag((I+Q)*V')
end

##################################
# integration
##################################

# given a vector whose t-th entry is a vector of intervals over which the t-th coordinate must be integrated
# returns the unique rectangular domains of integration composing the original domain of integration
# it return a vector with 1 rectangle per entry, defined by a tuple of all the lower bounds, followed by all the upper bounds 
function compute_boxes(intervals)
    boxes = Iterators.product(intervals...) |> collect
    boxes = vcat(boxes...)
    return map(boxes) do box
        ([getindex.(box, 1)...], [getindex.(box, 2)...])
    end
end

##################################
# dynamical rules
##################################

# returns a function x -> the boundaries of integration over h associated with x
function MX_constraints(dyn_params, epsilon)
    ne = sqrt_nu_eps(epsilon)

    kappa_low = dyn_params[1] / ne
    kappa_high = dyn_params[2] / ne
    @assert kappa_low <= kappa_high

    let kappa_low=kappa_low, kappa_high=kappa_high
        x -> begin
            T = length(x)-1
            [ if x[t] == 1 && x[t+1] == 1
                    isinf(kappa_high) ? [[-kappa_low, Inf]] : [[-Inf, -kappa_high], [-kappa_low, Inf]]
                elseif x[t] == 1 && x[t+1] == -1
                    [[-kappa_high, -kappa_low]]
                elseif x[t] == -1 && x[t+1] == 1
                    [[kappa_low, kappa_high]]
                elseif x[t] == -1 && x[t+1] == -1
                    isinf(kappa_high) ? [[-Inf, kappa_low]] : [[-Inf, kappa_low], [kappa_high, Inf]]
            end for t in 1:T]
        end
    end
end

# returns the function sigma(x,h) of the mixed RL/EO dynamics
function MX_dynamics(dyn_params) 
    kappa_low = dyn_params[1]
    kappa_high = dyn_params[2]
    @assert kappa_low <= kappa_high

    f = let kappa_low=kappa_low, kappa_high=kappa_high
        (x,h) -> if kappa_low < - x * h < kappa_high
                return -x
            else
                return x
            end
        end
    f
end


