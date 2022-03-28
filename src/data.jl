function load_gen_output(datasets, path_output, path_h5, n_params, n_particles, n_samples)
    fit_results = Dict()
    incomplete_datasets = Dict()

    for dataset=datasets
        data = import_data(joinpath(path_h5, "$(dataset)-data.h5"))
        if haskey(data, "stim_begin_confocal")
            stim = data["stim_begin_confocal"][1]
            ranges = [1:Int(stim-1), Int(stim+10):800, 801:1200, 1201:1600]
        else
            ranges = [1:400, 401:800, 801:1200, 1201:1600]
        end

        n_neurons = size(data["trace_array"], 1)
        fit_results[dataset] = Dict()
        fit_results[dataset]["v"] = data["velocity"]
        fit_results[dataset]["θh"] = data["θh"]
        fit_results[dataset]["P"] = data["pumping"]
        fit_results[dataset]["ranges"] = ranges
        fit_results[dataset]["num_neurons"] = n_neurons
        fit_results[dataset]["trace_params"] = zeros(length(ranges), n_neurons, n_particles, n_params)
        fit_results[dataset]["log_weights"] = zeros(length(ranges), n_neurons, n_particles)
        fit_results[dataset]["trace_scores"] = zeros(length(ranges), n_neurons, n_particles)
        fit_results[dataset]["sampled_trace_params"] = zeros(length(ranges), n_neurons, n_samples, n_params)
        fit_results[dataset]["log_ml_est"] = zeros(length(ranges), n_neurons)
        
        incomplete_datasets[dataset] = zeros(Bool, length(ranges), n_neurons)
        for (i,rng)=enumerate(ranges)
            for neuron = 1:n_neurons
                try
                    h5open(joinpath(path_output, "$(dataset)/$(Int(rng[1]))to$(Int(rng[end]))/h5/$(neuron).h5")) do f
                        fit_results[dataset]["trace_params"][i,neuron,:,:] .= read(f, "trace_params")
                        fit_results[dataset]["log_weights"][i,neuron,:] .= read(f, "log_weights")
                        fit_results[dataset]["trace_scores"][i,neuron,:] .= read(f, "trace_scores")
                        fit_results[dataset]["sampled_trace_params"][i,neuron,:,:] .= read(f, "sampled_trace_params")
                        fit_results[dataset]["log_ml_est"][i,neuron] = read(f, "log_ml_est")
                    end
                catch e
    #                 println(e)
                    incomplete_datasets[dataset][i,neuron] = true
                end
            end
        end
    end
    return fit_results, incomplete_datasets
end

