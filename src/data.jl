"""
Loads Gen output data.

# Arguments:
- `datasets`: Datasets to load
- `path_output`: Path to Gen output. Data must be stored in `path_output/dataset/rng1torng2/h5/neuron.h5`
- `path_h5`: Path to H5 file for the dataset, which must be stored in `dataset-data.h5` in this directory.
- `n_params`: Number of parameters in the Gen model
- `n_particles`: Number of particles in the Gen fit
- `n_samples`: Number of samples from the posterior given by the Gen fit
- `is_mcmc`: Whether fits are done via MCMC (as opposed to SMC)
"""
function load_gen_output(datasets, fit_ranges, path_output, path_h5, n_params, n_particles, n_samples, is_mcmc)
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

function get_pumping_ranges(datasets, P_ranges; rngs_valid=[5,6])
    rngs = Dict()
    for dataset in datasets
        rngs[dataset] = argmax([P_ranges[dataset][r][2] - P_ranges[dataset][r][1] for r=rngs_valid]) + rngs_valid[1] - 1
    end
    return rngs
end
