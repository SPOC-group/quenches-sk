include("../dmftSK/main.jl")

# using DataFrames, CSV, JLD2

databk = load("raw_data/data_bk_raw.jld2")["data"]
datafd = load("raw_data/data_fd_raw.jld2")["data"]

#====================================
    backtracking
====================================#

# clean points at c=2 for which a point at c=1 with larger entropy is present, or if their energy is too large (meaning that the c=1 fixed point was found)
databk = map(eachrow(databk)) do row
    if isnan(row.norm_entropy) 
        nothing
    elseif row.c == 2 && row.p == 7 .&& row.keo == 0.185 # point for which I only have runs at c=2 that fall back on c=1, even though at 0.19 I have a run at c=2 which is non-trivial
        nothing
    else
        row
    end
end |> v -> filter!(!isnothing, v) |> DataFrame

# for each p,c,bc,keo,krl take largest entropy
databk = combine(groupby(databk, [:p, :c, :bc, :keo, :krl])) do group
    i = argmax(group.norm_entropy)
    group[i, :]
end


#====================================
    backtracking sync-gr
====================================#

keo_max = [0.45, 0.38, 0.30, 0.27, 0.24, 0.22, 0.21, 0.19]

data = databk[databk.krl .== Inf .&& databk.p .<= 7, :] 
sort!(data, [:keo, :bc, :c, :p])

out = []
for p in 0:7
    i = p+1

    # c=1, bc=1,-1
    db1 = data[data.p .== p .&& data.c .== 1 .&& data.bc .== 1 .&& 0 .<= data.keo .<= 1, :]
    db2 = data[data.p .== p .&& data.c .== 1 .&& data.bc .== -1 .&& 0 .< data.keo .<= 1, :]

    ks = vcat(-reverse(db2.keo), db1.keo)
    ss = vcat(reverse(db2.norm_entropy), db1.norm_entropy)
    es = -vcat(-reverse(last.(db2.energy)), last.(db1.energy))

    # c=1, bc=1
    for i in 1:length(ks)

        push!(out, (
                p=p,
                c=1,
                bc=1,
                keo = ks[i],
                norm_entropy = ss[i],
                energy = es[i]
            ))

        # c=1, bc=-1
        push!(out, (
            p=p,
            c=1,
            bc=-1,
            keo = reverse(-ks)[i],
            norm_entropy = reverse(ss)[i],
            energy = -reverse(es)[i]
        ))
    end

    # c=2, bc=1
    db1 = data[data.p .== p .&& data.c .== 2 .&& data.bc .== 1 .&& 0 .<= data.keo .<= keo_max[i], :]

    ks = db1.keo
    ks = vcat(-reverse(ks[2:end]), ks)

    ss = db1.norm_entropy
    ss = vcat(reverse(ss[2:end]), ss)

    es = map(db1.energy) do e
        (e[end-1] + e[end])/2
    end
    es = -vcat(-reverse(es[2:end]), es)

    # c=2, bc=1
    for i in 1:length(ks)
        push!(out, (
            p=p,
            c=2,
            bc=1,
            keo = ks[i],
            norm_entropy = ss[i],
            energy = es[i]
        ))
    end

end

out = DataFrame(out)
CSV.write("data/data_syncGR_BK.csv", out)

#====================================
    backtracking, energy, RL
====================================#
data = databk
p_max = 5
krl_min = [0.78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


data = data[data.keo .== 0 .&& data.krl .< Inf .&& databk.bc .== 1, :]
data = data[data.p .<= p_max, :]
sort!(data, :krl)   
out = []

for p in 0:p_max
    i = p+1

    db1 = data[data.p .== p .&& data.c .== 1, :]
    push!(out, (p=p, c=1, krl=db1.krl, energy = - maximum.(db1.energy)))


    db1 = data[data.p .== p .&& data.c .== 2 .&& data.krl .>= krl_min[i], :]
    push!(out, (p=p, c=2, krl=db1.krl, energy = - maximum.(db1.energy)))

end

out = DataFrame(out) 
CSV.write("data/bk_rl_energy.csv", out)

#====================================
    backtracking, entropy, RL
====================================#
data = databk
p_max = 5
krl_min = [0.78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

data = data[data.keo .== 0 .&& data.krl .< Inf .&& databk.bc .== 1, :]
data = data[data.p .<= p_max, :]
sort!(data, :krl)   
out = []

for p in 0:p_max
    i = p+1

    db1 = data[data.p .== p .&& data.c .== 1 .&& data.bc .== 1, :]
    push!(out, (p=p, c=1, krl=db1.krl, norm_entropy=db1.norm_entropy))

    db1 = data[data.p .== p .&& data.c .== 2  .&& data.bc .== 1 .&& data.krl .>= krl_min[i], :]
    push!(out, (p=p, c=2, krl=db1.krl, norm_entropy=db1.norm_entropy))

end

out = DataFrame(out) 
CSV.write("data/bk_rl_entropy.csv", out)

#====================================
    fit sync-gr at keo = 0, krl = 0 of (p,c=1) to E_gs
====================================#
data = databk
data = data[data.keo .== 0 .&& data.krl .== Inf .&& data.c .== 1 .&& data.bc .== 1, :]
data.max_energy = maximum.(data.energy)
data = combine(groupby(data, :p), [:max_energy, :norm_entropy] .=> mean .=>  [:max_energy, :norm_entropy] )
CSV.write("data/fit_e_vs_p.csv", data)

#====================================
    forward
====================================#

#====================================
    energy, sync-gr
====================================#
data = datafd
data = data[data.krl .== Inf, :]
sort!(data, :keo)

ks = data.keo
es = data.energy
maxt = maximum(length.(es))
es = map(es, ks) do e, k
    if k > 0.1 || k == 0
        vcat(e, repeat([maximum(e)], maxt-length(e)))
    else
        vcat(e, repeat([NaN], maxt-length(e)))
    end
    # vcat(e, repeat([NaN], maxt-length(e)))
end
es = hcat(es...)'

data = hcat(ks, [row for row in eachrow(es)])
data = DataFrame(data, [:keo, :energy])
CSV.write("data/fd_eo_energy.csv", data)

#====================================
    energy, sync-rl
====================================#

data = datafd
data = data[data.keo .== 0, :]
sort!(data, :krl)
data = data[:, [:krl, :energy]]
CSV.write("data/fd_rl_energy.csv", data)

#====================================
    end time correlator, sync-gr
====================================#

data = datafd
data = data[data.krl .== Inf, :]
sort!(data, :keo)

ks = data.keo
maxt = maximum(length.(data.energy))
corr_endend = map(p -> fd_fromParamsToMatrices(p)[1][end-1,end], data.params)

data = hcat(ks, corr_endend)
data = DataFrame(data, [:keo, :Qendend])
CSV.write("data/fd_eo_correlation.csv", data)

#====================================
    field distribution
====================================#

#====================================
    pdf xh, sync-gr   # takes order 1h on m1 pro
====================================#

ps = 0:10
ys = [
    begin
        println(p)   
        dx=1e-3
        dbk = databk[databk.p .== p .&& databk.c .== 1 .&& databk.krl .== Inf .&& databk.keo .== 0  .&& databk.bc .== 1, :]
        p, c, keo, params = dbk[1, [:p, :c, :keo, :params]]
        dyn_params = [1e-16, Inf]
        t=p+c

        y1 = bk_fields_distribution_at_xht(0., t, p, c, dyn_params, params; epsilon = 0.)
        y2 = bk_fields_distribution_at_xht(0. +dx, t, p, c, dyn_params, params; epsilon = 0.)

        z1 = bk_fields_distribution_at_ht(1e-8, t, p, c, dyn_params, params; epsilon = 0.)
        z2 = bk_fields_distribution_at_ht(1e-8 + dx, t, p, c, dyn_params, params; epsilon = 0.)

        xhs = p <= 6 ? (0.:0.02:0.5) :  (0.:0.05:0.5)
        vs = [ bk_fields_distribution_at_xht(xh, t, p, c, dyn_params, params; epsilon = 0.) for xh in xhs]
        (y1, (y2-y1)/dx, z1, (z2-z1)/dx, collect(xhs), vs)
    end
    for p in ps 
]  

intercept_xh    = getindex.(ys, 1)
slope_xh        = getindex.(ys, 2)
intercept_h    = getindex.(ys, 3)
slope_h        = getindex.(ys, 4)
distribution_xhs = getindex.(ys, 5)
distribution_p = getindex.(ys, 6)

db = DataFrame(p = ps, c = ones(Int, length(ps)), keo = zeros(length(ps)), 
    pdf_xh_zero_intercept = intercept_xh, pdf_xh_zero_slope = slope_xh,
    pdf_h_zero_intercept = intercept_h, pdf_h_zero_slope = slope_h,
    pdf_xh_values = distribution_xhs, pdf_y_values = distribution_p
    )
CSV.write("data/bk_pdfxh_c1_keo0.csv", db)


#====================================
    pdf xh, sync-gr
====================================#

c=1
krl = Inf
keo = 0.1
plt = plot(ylim = (0, :auto))

out = []
ys = [
    begin     
        println(p)   
        dbk = databk[databk.p .== p .&& databk.c .== 1 .&& databk.krl .== krl .&& databk.keo .== keo  .&& databk.bc .== 1, :]
        p, params = dbk[1, [:p, :params]]
        dyn_params = [keo, krl]
        t=p+c
        xhs = vcat(-keo-0.2, -keo-1e-5, -keo:0.05:5)
        n = 2 * bk_Z(p, c, params, MX_constraints(dyn_params, 0), Random.default_rng())  
        ys = map(h -> bk_fields_distribution_at_xht(h, t, p, c, dyn_params, params, normalisation = n), xhs)
        push!(out, (p = p, xhs = xhs, p_xhs = ys))
    end
    for p in 7:7
]  
display(plt)

db = DataFrame(out)
CSV.write(string("distribution_fitnesses_xh_c1_kgr", keo, ".csv"), db)

