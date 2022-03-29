"""
Makes a distance matrix (compatible with t-SNE) of neurons between multiple datasets.

# Arguments:
- `datasets`: Array of datasets to use.
- `fit_results`: Gen fit results
- `v_ranges`: Range on velocity behavior
- `θh_ranges`: Range on head angle behavior
- `P_ranges`: Range on pumping behavior
- `neuron_categorization`: Categorization of neurons from `categorize_all_neurons` method
- `s_weight` (optional, default `1`): Weight of EWMA decay time `s`
- `v_weight` (optional, default `1`): Weight of velocity variable relative to other variables
- `min_P_rng` (optional, default `[0,2]`): Reject datasets and time ranges where pumping activity (25,75)th percentile doesn't include this range
- `use_all_ranges` (optional, default `false`): Use all avaialble ranges (vs select best range per dataset). 
    If `true`, could result in unequal number of ranges between datasets.
- `n_particles` (optional, default `2048`): Number of particles per neuron in the Gen fits
"""
function make_distance_matrix(datasets, fit_results, v_ranges, θh_ranges, P_ranges, neuron_categorization; s_weight=1, v_weight=1, min_P_rng=[0,2], use_all_ranges=false, n_particles=2048)
    rngs_use = Dict()
    idx_arr = Dict()
    idx_last = 0
    v_range = [-Inf,0,0,Inf]
    θh_range = [-Inf,Inf]
    P_range = [-Inf,Inf]
    for (i, dataset) = enumerate(datasets)
        rng_use = []
        rng_qual = []
        for (j,rng) = enumerate(fit_results[dataset]["ranges"])
            if P_ranges[dataset][j][1] <= min_P_rng[1] && P_ranges[dataset][j][2] >= min_P_rng[2]
                push!(rng_use, j)
                push!(rng_qual, P_ranges[dataset][j][2] - P_ranges[dataset][j][1])
            end
        end
        if isempty(rng_use)
            @warn("Dataset $(dataset) has no pumping-usable ranges")
            continue
        end
        if use_all_ranges
            rngs_use[dataset] = rng_use
        else
            rngs_use[dataset] = [rng_use[argmax(rng_qual)]]
        end

        idx_arr[dataset] = []
        for rng in rngs_use[dataset]
            push!(idx_arr[dataset], idx_last)
            idx_last += length(neuron_categorization[dataset][rng]["all"])
            v_range[1] = max(v_range[1], v_ranges[dataset][rng][1])
            v_range[2] = min(v_range[2], v_ranges[dataset][rng][2])
            θh_range[1] = max(θh_range[1], θh_ranges[dataset][rng][1])
            θh_range[2] = min(θh_range[2], θh_ranges[dataset][rng][2])
            P_range[1] = max(P_range[1], P_ranges[dataset][rng][1])
            P_range[2] = min(P_range[2], P_ranges[dataset][rng][2])
        end
    end

    distance_matrix = zeros(idx_last, idx_last)
    dataset_ids = fill("", idx_last)
    rng_ids = zeros(idx_last)
    neuron_ids = zeros(idx_last)

    deconvolved_activities = zeros(idx_last, n_particles, length(v_range), length(θh_range), length(P_range))
    
    for dataset = datasets
        if !haskey(rngs_use, dataset)
            continue
        end
        for (rng_idx,rng) = enumerate(rngs_use[dataset])
            offset1 = 0
            n_encoding = neuron_categorization[dataset][rng]["all"]
            for n1 = 1:fit_results[dataset]["num_neurons"]
                if !(n1 in n_encoding)
                    continue
                end
                idx1 = sum(n_encoding .<= n1) + idx_arr[dataset][rng_idx]
                dataset_ids[idx1] = dataset
                rng_ids[idx1] = rng
                neuron_ids[idx1] = n1
                deconvolved_activities[idx1,:,:,:,:] .= get_deconvolved_activity(fit_results[dataset]["sampled_trace_params"][rng,n1,:,:], v_range, θh_range, P_range)
            end

            for n1 = 1:fit_results[dataset]["num_neurons"]-1
                if !(n1 in n_encoding)
                    continue
                end
                idx1 = sum(n_encoding .<= n1) + idx_arr[dataset][rng_idx]

                for n2 = n1+1:fit_results[dataset]["num_neurons"]
                    if !(n2 in n_encoding)
                        continue
                    end
                    idx2 = sum(n_encoding .<= n2) + idx_arr[dataset][rng_idx]

                    n_cat = neuron_p_vals(deconvolved_activities[idx1,:,:,:,:], deconvolved_activities[idx2,:,:,:,:], compute_p=false)
                    
                    distance_matrix[idx1,idx2] = v_weight * sum([sum(n_cat["v_encoding"][i,i+1,:,:]) for i=1:3])
                    distance_matrix[idx1,idx2] += 4 * sum(n_cat["rev_θh_encoding"])
                    distance_matrix[idx1,idx2] += 4 * sum(n_cat["fwd_θh_encoding"])
                    distance_matrix[idx1,idx2] += 4 * sum(n_cat["rev_P_encoding"])
                    distance_matrix[idx1,idx2] += 4 * sum(n_cat["fwd_P_encoding"])
                    distance_matrix[idx1,idx2] += s_weight * abs(median(compute_s.(fit_results[dataset]["sampled_trace_params"][rng,n1,:,7]))
                            - median(compute_s.(fit_results[dataset]["sampled_trace_params"][rng,n2,:,7])))
                    distance_matrix[idx2,idx1] = distance_matrix[idx1,idx2]
                end
            end
        end
    end
    return distance_matrix, deconvolved_activities, dataset_ids, rng_ids, neuron_ids
end