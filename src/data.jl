"""
Loads Gen output data.

# Arguments:
- `datasets::Vector{String}`: Datasets to load
- `fit_ranges`: Dictionary of fit ranges for each dataset
- `path_output`: Path to Gen output. Data must be stored in `path_output/dataset/rng1torng2/h5/neuron.h5`
- `path_h5`: Path to H5 file for the dataset, which must be stored in `dataset-data.h5` in this directory.
- `n_params`: Number of parameters in the Gen model
- `n_particles`: Number of particles in the Gen fit
- `n_samples`: Number of samples from the posterior given by the Gen fit
- `is_mcmc`: Whether fits are done via MCMC (as opposed to SMC)
"""
function load_CePNEM_output(datasets::Vector{String}, fit_ranges::Dict, path_output, path_h5, n_params, n_particles, n_samples, is_mcmc)
    fit_results = Dict()
    incomplete_datasets = Dict()

    @showprogress for dataset=datasets
        data = import_data(joinpath(path_h5, "$(dataset)-data.h5"))
        ranges = fit_ranges[dataset]
        n_t = data["n_t"]
        n_neurons = size(data["trace_array"], 1)
        fit_results[dataset] = Dict()
        fit_results[dataset]["v"] = data["velocity"]
        fit_results[dataset]["θh"] = data["θh"]
        fit_results[dataset]["P"] = data["pumping"]
        fit_results[dataset]["ang_vel"] = data["ang_vel"]
        fit_results[dataset]["curve"] = data["curve"]
        fit_results[dataset]["trace_array"] = data["trace_array"]
        fit_results[dataset]["trace_original"] = data["trace_original"]
        fit_results[dataset]["ranges"] = ranges
        fit_results[dataset]["num_neurons"] = n_neurons
        if !is_mcmc
            fit_results[dataset]["trace_params"] = zeros(length(ranges), n_neurons, n_particles, n_params)
            fit_results[dataset]["log_weights"] = zeros(length(ranges), n_neurons, n_particles)
            fit_results[dataset]["trace_scores"] = zeros(length(ranges), n_neurons, n_particles)
            fit_results[dataset]["log_ml_est"] = zeros(length(ranges), n_neurons)
        end

        fit_results[dataset]["sampled_trace_params"] = zeros(length(ranges), n_neurons, n_samples, n_params)
        fit_results[dataset]["sampled_tau_vals"] = zeros(length(ranges), n_neurons, n_samples)

        list_t_confocal = data["timestamp_confocal"]
        if n_t > 800
            fit_results[dataset]["avg_timestep"] = (mean(diff(list_t_confocal[1:800])) +
                mean(diff(list_t_confocal[801:n_t]))) / 2
        else
            fit_results[dataset]["avg_timestep"] = mean(diff(list_t_confocal[1:n_t]))
        end

        incomplete_datasets[dataset] = zeros(Bool, length(ranges), n_neurons)
        for (i,rng)=enumerate(ranges)
            for neuron = 1:n_neurons
                try
                    h5open(joinpath(path_output, "$(dataset)/$(Int(rng[1]))to$(Int(rng[end]))/h5/$(neuron).h5")) do f
                        if !is_mcmc
                            fit_results[dataset]["trace_params"][i,neuron,:,:] .= read(f, "trace_params")
                            fit_results[dataset]["log_weights"][i,neuron,:] .= read(f, "log_weights")
                            fit_results[dataset]["trace_scores"][i,neuron,:] .= read(f, "trace_scores")
                            fit_results[dataset]["log_ml_est"][i,neuron] = read(f, "log_ml_est")
                        end
                        fit_results[dataset]["sampled_trace_params"][i,neuron,:,:] .= read(f, "sampled_trace_params")
                        s = compute_s.(fit_results[dataset]["sampled_trace_params"][i,neuron,:,7])
                        fit_results[dataset]["sampled_tau_vals"][i,neuron,:] .= log.(s ./ (s .+ 1), 0.5) .* fit_results[dataset]["avg_timestep"]
                    end
                catch e
                    incomplete_datasets[dataset][i,neuron] = true
                end
            end
        end
    end
    return fit_results, incomplete_datasets
end

"""
Computes time ranges with the most pumping variance for each dataset.

# Arguments:
- `datasets::Vector{String}`: List of datasets to analyze
- `P_ranges::Dict`: Dictionary of pumping ranges for each dataset
- `rngs_valid::Vector{Int}` (optional, default `[1,2]`): List of time ranges to consider
"""
function get_pumping_ranges(datasets::Vector{String}, P_ranges::Dict; rngs_valid::Vector{Int}=[1,2])
    rngs = Dict()
    for dataset in datasets
        rngs[dataset] = argmax([P_ranges[dataset][r][2] - P_ranges[dataset][r][1] for r=rngs_valid]) + rngs_valid[1] - 1
    end
    return rngs
end

"""
Adds a dictionary of new results to an existing analysis dictionary.

# Arguments:
- `analysis_dict::Dict`: Dictionary of CePNEM analysis results
- `new_dict::Dict`: Dictionary of new results to add
- `key::String`: Key in `analysis_dict` to add new results to
"""
function add_to_analysis_dict!(analysis_dict::Dict, new_dict::Dict, key::String)
    if !haskey(analysis_dict, key)
        analysis_dict[key] = Dict()
    end
    for dataset in keys(new_dict)
        analysis_dict[key][dataset] = new_dict[dataset]
    end
end

"""
Computes signal value for each neuron in each dataset.

# Arguments:
- `fit_results::Dict`: Dictionary of CePNEM fit results.
"""
function compute_signal(fit_results::Dict)
    signal = Dict()
    @showprogress for dataset in keys(fit_results)
        signal[dataset] = Dict()
        for neuron in 1:fit_results[dataset]["num_neurons"]
            trace_original = fit_results[dataset]["trace_original"][neuron,:]
            signal[dataset][neuron] = std(trace_original) / mean(trace_original)
        end
    end
    return signal
end

"""
Computes match and encoding-change match dictionaries for NeuroPAL data.

# Arguments:
- `fit_results::Dict}`: Dictionary of CePNEM fit results.
- `analysis_dict::Dict`: Dictionary of CePNEM analysis results.
- `list_class::Vector{Any}`: List of classes to match.
- `list_match_dict::Vector{Any}`: List of dictionaries of matches for each class.
- `datasets_neuropal::Vector{String}`: List of datasets to match.
"""
function neuropal_data_to_dict(fit_results::Dict, analysis_dict::Dict, list_class::Vector, list_match_dict::Vector, datasets_neuropal::Vector{String})
    matches = Dict()
    matches_ec = Dict()
    @showprogress for dataset = datasets_neuropal
        idx_uid = findall(x->x==dataset, datasets_neuropal)[1]
        match_roi_class, match_class_roi = list_match_dict[idx_uid]
        list_ec = Int32[]

        if length(fit_results[dataset]["ranges"]) > 1
            for rng1 in 1:length(fit_results[dataset]["ranges"])-1
                for rng2 in rng1+1:length(fit_results[dataset]["ranges"])
                    rng = (rng1, rng2)
                    push!(list_ec, analysis_dict["encoding_changing_neurons_msecorrect_mh"][dataset][rng]["neurons"])
                end
            end
        end
        for (_, match_neuron, _) = list_class
            if !(match_neuron in keys(match_class_roi))
                continue
            end

            for n_ in match_class_roi[match_neuron]
                n = n_[2]

                if !haskey(matches, match_neuron)
                    matches[match_neuron] = []
                end
                push!(matches[match_neuron], (dataset, n))

                if n in list_ec
                    if !haskey(matches_ec, match_neuron)
                        matches_ec[match_neuron] = []
                    end
                    push!(matches_ec[match_neuron], (dataset, n))
                end
            end
        end
    end
    return matches, matches_ec
end
