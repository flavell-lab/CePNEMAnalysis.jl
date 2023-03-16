"""
    DEPRECATED in favor of `compute_extrapolated_CePNEM_posterior_stats`.

    Extrapolates the model fits to the full range of behaviors defined by `analysis_dict["extrapolated_behaviors"]`.
    Computes the model fits for each point in the CePNEM posterior distribution, and computes the mean and variance of the extrapolated model fits at each time point.

    # Arguments:
    - `analysis_dict::Dict`: CePNEM analysis dictionary
    - `fit_results::Dict`: dictionary of CePNEM fit results
    - `neuron::String`: neuron label
    - `datasets_use::Ar::Vector{String}`: list of datasets to consider in the computation (should only include datasets from NeuroPAL animals)
    - `n_idx::Int` (optional, default `10001`): number of points in the CePNEM posterior distribution to use. If set to the total number of points, all will be used; otherwise, points will be randomly sampled.
    - `max_rng::Int` (optional, default `2`): maximum number of time ranges in a given dataset.

    # Returns:
    - `mean_extrap::Matrix{Float64}`: array of all mean extrapolated model fits, across datasets, time ranges, and L/R neuron identity
    - `var_extrap::Matrix{Float64}`: array of all variances of extrapolated model fits, across datasets, time ranges, and L/R neuron identity
    - `mean_extrap_dict::Dict`: dictionary of mean extrapolated model fits, with keys (dataset, range, neuron)
    - `var_extrap_dict::Dict`: dictionary of variance of extrapolated model fits, with keys (dataset, range, neuron)
"""
function compute_extrapolated_fits_meanstd(analysis_dict::Dict, fit_results::Dict, neuron::String, datasets_use::Vector{String}; n_idx::Int=10001, max_rng::Int=2)
    all_behs = analysis_dict["extrapolated_behaviors"]

    mean_extrap_dict = Dict()
    var_extrap_dict = Dict()
    mean_extrap = zeros(max_rng*length(analysis_dict["matches"][neuron]), size(all_behs,1))
    var_extrap = zeros(max_rng*length(analysis_dict["matches"][neuron]), size(all_behs,1))
    count = 1

    for (dataset, n) in analysis_dict["matches"][neuron]
        if !in(dataset, datasets_use)
            continue
        end
        for rng=1:length(fit_results[dataset]["ranges"])
            extrap = zeros(size(all_behs,1), n_idx)
            for idx = 1:n_idx
                if n_idx == size(fit_results[dataset]["sampled_trace_params"],3)
                    ps = deepcopy(fit_results[dataset]["sampled_trace_params"][rng,n,idx,1:8])
                else
                    ps = deepcopy(fit_results[dataset]["sampled_trace_params"][rng,n,rand(1:size(fit_results[dataset]["sampled_trace_params"],3)),1:8])
                end
                ps = deepcopy(fit_results[dataset]["sampled_trace_params"][rng,n,idx,1:8])
                ps[3] = ps[3] * (2*θh_pos_is_ventral[dataset]-1)
                ps[6] = 0
                model = model_nl8(size(all_behs,1), ps..., all_behs[:,1], all_behs[:,2], all_behs[:,3])
                extrap[:,idx] .= (model .- mean(model)) .* analysis_dict["signal"][dataset][n]
            end
            mean_extrap[count,:] .= mean(extrap, dims=2)[:,1]
            mean_extrap_dict[(dataset, rng, n)] = mean_extrap[count,:]
            var_extrap[count,:] .= var(extrap, dims=2)[:,1]
            var_extrap_dict[(dataset, rng, n)] = var_extrap[count,:]
            
            count += 1
        end
    end

    mean_extrap = mean_extrap[1:count-1, :]
    var_extrap = var_extrap[1:count-1, :]

    return mean_extrap, var_extrap, mean_extrap_dict, var_extrap_dict
end

"""
    Sorts CePNEM extrapolated model fits by which neuron they correspond to.

    # Arguments:
    - `analysis_dict::Dict`: CePNEM analysis dictionary
    - `mean_extrap_dict_all::Dict`: dictionary of mean extrapolated model fits, with keys (dataset, range, neuron)
    - `var_extrap_dict_all::Dict`: dictionary of variance of extrapolated model fits, with keys (dataset, range, neuron)
    - `datasets_use::Union{Nothing,Vector{String}}` (optional, default `nothing`): list of datasets to consider in the computation (should only include datasets from NeuroPAL animals). Use all if left at default.
    - `rngs_use::Union{Nothing,Vector{Int}}` (optional, default `nothing`): list of time ranges to consider in the computation. Use all if left at default.
"""
function neuropal_sort_extrapolated_fits(analysis_dict::Dict, mean_extrap_dict_all::Dict, var_extrap_dict_all::Dict; datasets_use::Union{Nothing,Vector{String}}=nothing, rngs_use::Union{Nothing,Vector{Int}}=nothing)    
    mean_extrap_dict = Dict()
    var_extrap_dict = Dict()

    for neuron in keys(analysis_dict["matches"])
        mean_extrap_dict[neuron] = Dict()
        var_extrap_dict[neuron] = Dict()
        for (dataset, n) in analysis_dict["matches"][neuron]
            if !isnothing(datasets_use) && !in(dataset, datasets_use)
                continue
            end
            for rng=1:size(mean_extrap_dict_all[dataset],1)
                if !isnothing(rngs_use) && !in(rng, rngs_use[dataset])
                    continue
                end
                mean_extrap_dict[neuron][(dataset, rng, n)] = mean_extrap_dict_all[dataset][rng,n,:]
                var_extrap_dict[neuron][(dataset, rng, n)] = var_extrap_dict_all[dataset][rng,n,:]
            end
        end
    end
    return mean_extrap_dict, var_extrap_dict
end

"""
    Converts a dictionary of extrapolated model fits to an array, with each row corresponding to a dataset/time range/neuron identity combination.

    # Arguments:
    - `extrap_dict::Dict`: dictionary of extrapolated model fits, with keys (dataset, range, neuron)
    - `datasets_use::Union{Nothing,Vector{String}}` (optional, default `nothing`): list of datasets to consider in the computation (should only include datasets from NeuroPAL animals). Use all if left at default.
    - `ranges_use::Union{Nothing,Dict{String,Vector{Int}}}` (optional, default `nothing`): dictionary of time ranges to consider in the computation, with keys (dataset) and values (list of time ranges). Use all if left at default.
"""
function extrap_dict_to_array(extrap_dict::Dict; datasets_use::Union{Nothing,Vector{String}}=nothing, ranges_use::Union{Nothing,Dict{String,Vector{Int}}}=nothing)
    arr = zeros(length(keys(extrap_dict)), length(collect(values(extrap_dict))[1]))
    count = 1
    for k in keys(extrap_dict)
        if length(k) == 3
            if (isnothing(datasets_use) || k[1] in datasets_use) && (isnothing(ranges_use) || k[2] in ranges_use[k[1]])
                arr[count,:] .= extrap_dict[k]
                count += 1
            end
        elseif length(k) == 2 # assume no range information present
            if (isnothing(datasets_use) || k[1] in datasets_use)
                arr[count,:] .= extrap_dict[k]
                count += 1
            end
        end
    end
    arr = arr[1:count-1, :]
    return arr
end

"""
    Computes the z-score of the mean extrapolated model fits, and the corresponding variance of the z-scored extrapolated model fits, weighted by the variance of the extrapolated model fits.

    # Arguments:
    - `mean_extrap_dict::Dict`: dictionary of mean extrapolated model fits, with keys (dataset, range, neuron)
    - `var_extrap_dict::Dict`: dictionary of variance of extrapolated model fits, with keys (dataset, range, neuron)
"""
function weighted_zscore(mean_extrap_dict::Dict, var_extrap_dict::Dict)
    zscored_mean = Dict()
    zscored_var = Dict()
    for (k,v) = mean_extrap_dict
        weights = AnalyticWeights(1 ./ var_extrap_dict[k])
        weighted_mean_extrap, weighted_var_extrap = mean_and_var(v, weights, corrected=true)
        
        zscored_mean[k] = (v .- weighted_mean_extrap) ./ sqrt(weighted_var_extrap)
        zscored_var[k] = var_extrap_dict[k] ./ weighted_var_extrap
    end
    return zscored_mean, zscored_var
end

"""
Subtracts the baseline from the mean extrapolated model fits. The baseline is computed as the
median of the extrapolated model fits over all time points where the worm didn't pump.

# Arguments:
- `analysis_dict::Dict`: CePNEM analysis dictionary
- `mean_extrap::Dict`: mean extrapolated model fits for all neurons
- `neuron::String`: neuron label
- `datasets_use::Vector{String}`: list of datasets to use
- `delay::Int` (optional, default `20`): number of time points to use for computing whether pumping is 0
- `P_thresh::Real` (optional, default `0.5`): threshold for pumping to be considered 0
"""
function baseline_correct_mean_extrap(analysis_dict::Dict, mean_extrap::Dict, neuron::String, datasets_use::Vector{String}; delay::Int=20, P_thresh::Real=0.5)
    baseline = zeros(size(mean_extrap[neuron], 1))
    all_behs = analysis_dict["extrapolated_behaviors"]

    count = 1
    for (dataset, n) in analysis_dict["matches"][neuron]
        if !in(dataset, datasets_use)
            continue
        end
        # many model fits fail to constrain pumping, so only apply baseline correction over non-pumping timepoints
        timepts_valid = [t for t in 1:size(mean_extrap[neuron], 2) if mean(all_behs[max(1,t-delay+1):t,3]) < P_thresh]
        baseline[count] = median(mean_extrap[neuron][count, timepts_valid])
        count += 1
    end
    return mean_extrap[neuron] .- baseline
end

"""
    Computes the variability index of a neuron, which is an approximation for how variable the neuron's tuning to behavior under CePNEM is across datasets and time ranges.

    # Arguments:
    - `mean_extrap::Matrix{Float64}`: mean extrapolated model fits
    - `var_extrap::Matrix{Float64}`: variance of extrapolated model fits
    - `median_std_extrap::Union{Nothing,Float64}` (optional, default `nothing`): median standard deviation of extrapolated model fits, computed over all datasets and time ranges. Set to `nothing` to compute this value automatically.
    - `use_mean_abs::Bool` (optional, default `false`): if true, use the mean absolute value of the extrapolated model fits instead of the mean
    - `subtract_gen_variability::Bool` (optional, default `false`): if true, subtract the CePNEM-posterior-based variability of the neuron across datasets and time ranges.
        **WARNING:** this relies on the assumption that independent CePNEM runs will produce distributions centered at different points in the posterior, and the distribution of these center points
        will resemble the posterior distribution. This assumption has been verified to be false in some situations, for instance if the neuron is not constrained to pump in the CePNEM posterior, and can result in negative variability in those instances.
        Therefore, this option should be used with caution.
""" 
function compute_neuron_variability(mean_extrap::Matrix{Float64}, var_extrap::Matrix{Float64}; median_std_extrap::Union{Nothing,Float64}=nothing, use_mean_abs::Bool=false, subtract_gen_variability::Bool=false)
    weighted_mean_extrap = zeros(size(mean_extrap, 2))
    weighted_mean_abs_extrap = zeros(size(mean_extrap, 2))
    weighted_var_extrap = zeros(size(var_extrap, 2))
    weighted_mean_var = zeros(size(var_extrap, 2))
    variability_index = zeros(size(var_extrap, 2))
    for t=1:size(mean_extrap, 2)
        weights = AnalyticWeights(1 ./ var_extrap[:,t])
        weighted_mean_extrap[t], weighted_var_extrap[t] = mean_and_var(mean_extrap[:,t], weights, corrected=true)
        weighted_mean_abs_extrap[t] = mean(abs.(mean_extrap[:,t]), weights)
        # don't automatically correct since this is the analytical solution
        # instead, subtracting `1/sum(weights)` corrects for the fact that the variance is computed from the (inaccurate) `mean=0`
        weighted_mean_var[t] = var(sqrt.(max.(var_extrap[:,t] .- 1/sum(weights), 0)), weights, mean=0, corrected=true)
        diff_var = weighted_var_extrap[t] - subtract_gen_variability * weighted_mean_var[t]
        variability_index[t] = sqrt(abs(diff_var)) * sign(diff_var)
    end

    std_extrap = zeros(size(mean_extrap, 1))
    for i=1:size(mean_extrap, 1)
        std_extrap[i] = std(mean_extrap[i,:])
    end

    # TODO: mean or median for std_extrap?
    return (use_mean_abs ? median(weighted_mean_abs_extrap) : median(variability_index)) / (isnothing(median_std_extrap) ? median(std_extrap) : median_std_extrap), weighted_mean_extrap, weighted_var_extrap, weighted_mean_var, variability_index, std_extrap, weighted_mean_abs_extrap
end

"""
    Computes the variability of a neuron between time ranges within the same animal, thus giving a proxy for how much neurons change over the time course of our recordings, as opposed to between animals.

    # Arguments:
    - `fit_results::Dict`: dictionary of CePNEM fit results
    - `mean_extrap_dict::Dict`: dictionary of extrapolated mean model fits
    - `var_extrap_dict::Dict`: dictionary of extrapolated variance of model fits
    - `datasets_use::Union{Nothing,Vector{String}}` (optional, default `nothing`): if not `nothing`, only use the datasets in this vector
"""
function compute_intra_dataset_diffs(fit_results::Dict, mean_extrap_dict::Dict, var_extrap_dict::Dict; datasets_use::Union{Nothing,Vector{String}}=nothing)
    mean_extrap_diff = Dict()
    var_extrap_diff = Dict()

    for (dataset, rng, n) in keys(mean_extrap_dict)
        if !isnothing(datasets_use) && !in(dataset, datasets_use)
            continue
        end
        if length(fit_results[dataset]["ranges"]) == 1
            continue
        end
        @assert(length(fit_results[dataset]["ranges"]) == 2)
        if rng == 2
            continue
        end
        mean_extrap_diff[(dataset, n)] = (mean_extrap_dict[(dataset, 2, n)] - mean_extrap_dict[(dataset, 1, n)]) / sqrt(2)
        var_extrap_diff[(dataset, n)] = (var_extrap_dict[(dataset, 2, n)] + var_extrap_dict[(dataset, 1, n)]) / sqrt(2)

    end
    return mean_extrap_diff, var_extrap_diff
end

"""
    Merges neuron detections from different time ranges in the same animal together, to give a sense of variability of the neuron between animals but not within the same animal.
    Note that this function does not merge L and R neurons together.

    # Arguments:
    - `fit_results::Dict`: dictionary of CePNEM fit results
    - `mean_extrap_dict::Dict`: dictionary of extrapolated mean model fits
    - `var_extrap_dict::Dict`: dictionary of extrapolated variance of model fits
    - `datasets_use::Union{Nothing,Vector{String}}` (optional, default `nothing`): if not `nothing`, only use the datasets in this vector
    - `ranges_use::Union{Nothing,Dict{String,Vector{Int}}}` (optional, default `nothing`): if not `nothing`, only use the ranges in this dictionary
"""
function merge_datasets(fit_results::Dict, mean_extrap_dict::Dict, var_extrap_dict::Dict; datasets_use::Union{Nothing,Vector{String}}=nothing, ranges_use::Union{Nothing,Dict{String,Vector{Int}}}=nothing)
    mean_extrap_merged = Dict()
    var_extrap_merged = Dict()
    for (dataset, rng, n) in keys(mean_extrap_dict)
        if (dataset, n) in keys(mean_extrap_merged) || (!isnothing(datasets_use) && !in(dataset, datasets_use)) || (!isnothing(ranges_use) && !in(rng, ranges_use[dataset]))
            continue
        end
        n_ranges = length(fit_results[dataset]["ranges"])
        if n_ranges == 1 || (!isnothing(ranges_use) && length(ranges_use[dataset]) == 1)
            mean_extrap_merged[(dataset, n)] = mean_extrap_dict[(dataset, rng, n)]
            var_extrap_merged[(dataset, n)] = mean_extrap_dict[(dataset, rng, n)]
            continue
        end
        mean_extrap_merged[(dataset, n)] = zeros(length(mean_extrap_dict[(dataset, rng, n)]))
        var_extrap_merged[(dataset, n)] = zeros(length(mean_extrap_dict[(dataset, rng, n)]))
        for t = 1:length(mean_extrap_dict[(dataset, rng, n)])
            rngs_valid = isnothing(ranges_use) ? (1:n_ranges) : ranges_use[dataset]
            weights = AnalyticWeights(1 ./ [var_extrap_dict[(dataset, rng_, n)][t] for rng_ = rngs_valid])
            mean_extrap_merged[(dataset, n)][t] = mean([mean_extrap_dict[(dataset, rng_, n)][t] for rng_ = rngs_valid], weights)
            var_extrap_merged[(dataset, n)][t] = 1 / sum(weights)
        end
    end
    return mean_extrap_merged, var_extrap_merged
end
    