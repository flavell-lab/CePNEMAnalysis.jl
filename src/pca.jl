function extrapolate_neurons(datasets, fit_results, neuron_categorization, n_neurons_tot, P_ranges; P_diff_thresh=0.5, rngs_valid=nothing)
    P_diff_thresh = 0.5
    all_params = zeros(4*n_neurons_tot, 11)
    ids = []
    all_behs = zeros(1600 * length(datasets_baseline), 5)
    idx = 1
    idx_beh = 1
    rngs_valid = [5,6]
    for dataset in datasets_baseline
        rng = argmax([P_ranges[dataset][r][2] - P_ranges[dataset][r][1] for r=rngs_valid]) + rngs_valid[1] - 1
        if P_ranges[dataset][rng][2] - P_ranges[dataset][rng][1] < P_diff_thresh
            @warn("Skipping $(dataset) due to insufficient pumping variance")
            continue
        end
        for n=1:fit_results[dataset]["num_neurons"]
            if n in neuron_categorization[dataset][rng]["all"]
                all_params[idx,:] .= dropdims(median(fit_results[dataset]["sampled_trace_params"][rng, n, :, :], dims=1), dims=1)
                push!(ids, (dataset, rng, n))
                idx += 1
            end
        end
        r = fit_results[dataset]["ranges"][rng]
        all_behs[idx_beh:idx_beh+length(r)-1, 1] .= fit_results[dataset]["v"][r]
        all_behs[idx_beh:idx_beh+length(r)-1, 2] .= fit_results[dataset]["θh"][r] .* (2*θh_pos_is_ventral[dataset]-1)
        all_behs[idx_beh:idx_beh+length(r)-1, 3] .= fit_results[dataset]["P"][r]
        all_behs[idx_beh:idx_beh+length(r)-1, 4] .= fit_results[dataset]["ang_vel"][r] .* (2*θh_pos_is_ventral[dataset]-1)
        all_behs[idx_beh:idx_beh+length(r)-1, 5] .= fit_results[dataset]["curve"][r]
        idx_beh += length(r)
    end
    all_params = all_params[1:idx-1,:]
    all_behs = all_behs[1:idx_beh-1,:]
    
    models = zeros(size(all_params,1), size(all_behs,1))
    for i=1:size(models,1)
        models[i,:] .= model_nl8(size(all_behs,1), all_params[i,1:8]..., all_behs[:,1], all_behs[:,2], all_behs[:,3])
    end
    return models, ids, all_behs, all_params, fit(PCA, models)
end

function make_distance_matrix(models)
    distance_matrix = zeros(size(models,1), size(models,1))
    norms = Dict()
    @showprogress for i=1:size(models,1)-1
        if !(i in keys(norms))
            norms[i] = sum(models[i,:] .^ 2)
        end
        for j=i+1:size(models,1)
            if !(j in keys(norms))
                norms[j] = sum(models[j,:] .^ 2)
            end
            distance_matrix[i,j] = sum(models[i,:] .* models[j,:]) / sqrt(norms[i] * norms[j])
            distance_matrix[j,i] = distance_matrix[i,j]
        end
    end
    return distance_matrix
end


