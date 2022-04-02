"""
Updates choice map `cmap` to contain parameters `x` and/or neural data `y`.
"""
function update_cmap!(cmap, x, y)
    if !isnothing(x)
        cmap[:c_vT] = x[1]
        cmap[:c_v] = x[2]
        cmap[:c_θh] = x[3]
        cmap[:c_P] = x[4]
        cmap[:c] = x[5]
        cmap[:y0] = x[6]
        cmap[:s0] = x[7]
        cmap[:b] = x[8]
        cmap[:σ0] = x[9]
    end
    if !isnothing(y)
        for (i,val) = enumerate(y)
            cmap[:chain => i => :y] = val
        end
    end
    return cmap
end

"""
Given parameters `x` (given as an array of the 9 parameters for `nl8`), observed data `y_cmap`
(neural data, given as a `choicemap)`, `args` (behavioral data),
and `log_ml_est` from a Gen inference run on `y, args`, computes `P(x | y, args)` under model `nl8`.
Updates `y_cmap` to contain `x` values as well.
"""
function evaluate_pdf_xgiveny!(x, y_cmap, args, log_ml_est)
    update_cmap!(y_cmap, x, nothing)
    return exp(BigFloat(Gen.assess(unfold_nl8, args, y_cmap)[1] - log_ml_est))
end

"""
Computes the distributions given by `P(x | y1)` and `P(x | y2)`, where `y1` is given by
`dataset1`, `rng_idx_1`, and `neuron1`, and `y2` is given by `dataset2`, `rng_idx_2`, and `neuron2`.
Uses `Gen` fits stored in `fit_results`, and fixes parameters in `param_fix` to the MLE when doing cross-dataset comparison.
Ie: when evaluating `P(x1 | y2)` for `x1` drawn from `Gen` samples for `P(x | y1)`, defines `x1' = x1, x1'[param_fix] = MLE(y2, param_fix)`
Then computes `P(x | y1)` and `P(x | y2)` across all values of `x` sampled from either posterior.
"""
function compute_dists(dataset1, rng_idx_1, neuron1, dataset2, rng_idx_2, neuron2, fit_results, params_fix, param_noise=9)
    path1 = "/data1/prj_kfc/data/processed_h5/$(dataset1)-data.h5"
    results1 = fit_results[dataset1]
    rng1 = results1["ranges"][rng_idx_1]
    dict1 = import_data(path1)
    v1 = dict1["velocity"][rng1]
    θh1 = dict1["θh"][rng1]
    P1 = dict1["pumping"][rng1]
    args1 = (rng1[end]-rng1[1]+1, v1, θh1, P1)
    y1 = dict1["trace_array"][neuron1, rng1]
    cmap1 = Gen.choicemap()
    update_cmap!(cmap1, nothing, y1)
    x1 = results1["trace_params"][rng_idx_1, neuron1, :, :]
    mle1 = results1["trace_params"][rng_idx_1, neuron1, argmax(results1["trace_scores"][rng_idx_1,neuron1,:]), :]

    
    path2 = "/data1/prj_kfc/data/processed_h5/$(dataset2)-data.h5"
    results2 = fit_results[dataset2]
    rng2 = results2["ranges"][rng_idx_2]
    dict2 = import_data(path2)
    v2 = dict2["velocity"][rng2]
    θh2 = dict2["θh"][rng2]
    P2 = dict2["pumping"][rng2]
    y2 = dict2["trace_array"][neuron2, rng2]
    args2 = (rng2[end]-rng2[1]+1, v2, θh2, P2)
    cmap2 = Gen.choicemap()
    update_cmap!(cmap2, nothing, y2)
    x2 = results2["trace_params"][rng_idx_2, neuron2, :, :]
    mle2 = results2["trace_params"][rng_idx_2, neuron2, argmax(results2["trace_scores"][rng_idx_2,neuron2,:]), :]
    
    
    p_1 = zeros(BigFloat, size(x1,1)+size(x2,1))
    p_2 = zeros(BigFloat, size(x1,1)+size(x2,1))
    for i in 1:size(x1,1)
        x1_trial = deepcopy(x1[i,:])
        for param in params_fix
            x1_trial[param] = mle1[param]
        end
        p_1[i] = evaluate_pdf_xgiveny!(x1_trial, cmap1, args1, results1["log_ml_est"][rng_idx_1, neuron1])
        
        for param in params_fix
            x1_trial[param] = mle2[param]
        end
        p_2[i] = evaluate_pdf_xgiveny!(x1_trial, cmap2, args2, results2["log_ml_est"][rng_idx_2, neuron2])
    end
    
    for i in 1:size(x2,1)
        j=i+size(x1,1)
        x2_trial = deepcopy(x2[i,:])
        for param in params_fix
            x2_trial[param] = mle1[param]
        end
        p_1[j] = evaluate_pdf_xgiveny!(x2_trial, cmap1, args1, results1["log_ml_est"][rng_idx_1, neuron1])
        
        for param in params_fix
            x2_trial[param] = mle2[param]
        end
        p_2[j] = evaluate_pdf_xgiveny!(x2_trial, cmap2, args2, results2["log_ml_est"][rng_idx_2, neuron2])
    end
    
    return p_1, p_2
end

"""
Computes KL divergence between two distributions (entered as samples) `p` and `q`
"""
function kl_div(p, q)
    kl_pq = mean(p .* log.(p ./ q))
    kl_qp = mean(q .* log.(q ./ p))
    return min([kl_pq, kl_qp])
end

"""
Computes overlap index between two distributions (entered as samples) `p` and `q`
"""
function overlap_index(p,q)
    @assert(length(p) == length(q))
    p_norm = sum(p)
    q_norm = sum(q)
    return sum([min(p[i] / p_norm, q[i] / q_norm) for i=1:length(p)])
end

"""
Computes `Prob(p>q)` for `p~P`, `q~Q`
Here `P` and `Q` are arrays of samples
"""
function prob_P_greater_Q(P,Q)
    p_greater_q = 0
    Q_const = NaN
    if minimum(Q) == maximum(Q)
        Q_const = minimum(Q)
    end
    for p in P
        if !isnan(Q_const)
            p_greater_q += sum(p > Q_const) .+ sum(p == Q_const) / 2
        else
            p_greater_q += sum(p .> Q) + sum(p .== Q) / 2
        end
    end
    if !isnan(Q_const)
        return p_greater_q / length(P)
    else
        return p_greater_q / (length(P) * length(Q))
    end
end
