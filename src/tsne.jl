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
function make_distance_matrix(datasets, fit_results, v_ranges, θh_ranges, P_ranges, neuron_categorization; s_weight=2, v_weight=1, min_P_rng=[0,2], use_all_ranges=false, rngs_valid=nothing)
    n_particles = size(fit_results[datasets[1]]["sampled_trace_params"], 3)
    rngs_use = Dict()
    idx_arr = Dict()
    idx_last = 0
    v_range = [-Inf,0,0,Inf]
    θh_range = [-Inf,Inf]
    P_range = [-Inf,Inf]
    @showprogress for (i, dataset) = enumerate(datasets)
        rng_use = []
        rng_qual = []
        rngs_valid_use = 1:length(fit_results[dataset]["ranges"])
        if !isnothing(rngs_valid)
            rngs_valid_use = rngs_valid
        end
        for (j,rng) = enumerate(rngs_valid_use)
            if P_ranges[dataset][rng][1] <= min_P_rng[1] && P_ranges[dataset][rng][2] >= min_P_rng[2]
                push!(rng_use, rng)
                push!(rng_qual, P_ranges[dataset][rng][2] - P_ranges[dataset][rng][1])
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

        println(rngs_use[dataset])
        idx_arr[dataset] = []
        for rng in rngs_use[dataset]
            push!(idx_arr[dataset], idx_last)
            idx_last += length(neuron_categorization[dataset][rng]["all"])
            v_range[1] = max(v_range[1], v_ranges[dataset][rng][1])
            v_range[2] = v_range[1] / 100
            v_range[4] = min(v_range[4], v_ranges[dataset][rng][4])
            v_range[3] = v_range[4] / 100
            θh_range[1] = max(θh_range[1], θh_ranges[dataset][rng][1])
            θh_range[2] = min(θh_range[2], θh_ranges[dataset][rng][2])
            P_range[1] = max(P_range[1], P_ranges[dataset][rng][1])
            P_range[2] = min(P_range[2], P_ranges[dataset][rng][2])
        end
    end

    distance_matrix = zeros(idx_last, idx_last)
    dataset_ids = fill("", idx_last)
    rng_ids = zeros(Int, idx_last)
    neuron_ids = zeros(Int, idx_last)

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
        end
    end
    for idx1=1:size(distance_matrix,1)-1
        for idx2=idx1+1:size(distance_matrix,1)
            n_cat = neuron_p_vals(deconvolved_activities[idx1,:,:,:,:], deconvolved_activities[idx2,:,:,:,:], 0, compute_p=false)

            distance_matrix[idx1,idx2] = v_weight * sum([sum(n_cat["v_encoding"][i,i+1,:,:]) for i=1:3]) * v_STD / (v_range[4] - v_range[1])
            distance_matrix[idx1,idx2] += 2 * sum(n_cat["rev_θh_encoding_act"]) * θh_STD / (θh_range[2] - θh_range[1])
            distance_matrix[idx1,idx2] += 2 * sum(n_cat["rev_θh_encoding_inh"]) * θh_STD / (θh_range[2] - θh_range[1])
            distance_matrix[idx1,idx2] += 2 * sum(n_cat["fwd_θh_encoding_act"]) * θh_STD / (θh_range[2] - θh_range[1])
            distance_matrix[idx1,idx2] += 2 * sum(n_cat["fwd_θh_encoding_inh"]) * θh_STD / (θh_range[2] - θh_range[1])
            distance_matrix[idx1,idx2] += 2 * sum(n_cat["rev_P_encoding_act"]) * P_STD / (P_range[2] - P_range[1])
            distance_matrix[idx1,idx2] += 2 * sum(n_cat["rev_P_encoding_inh"]) * P_STD / (P_range[2] - P_range[1])
            distance_matrix[idx1,idx2] += 2 * sum(n_cat["fwd_P_encoding_act"]) * P_STD / (P_range[2] - P_range[1])
            distance_matrix[idx1,idx2] += 2 * sum(n_cat["fwd_P_encoding_inh"]) * P_STD / (P_range[2] - P_range[1])
            distance_matrix[idx1,idx2] += s_weight * abs(median(fit_results[dataset_ids[idx1]]["sampled_tau_vals"][rng_ids[idx1],neuron_ids[idx1],:])
                    - median(fit_results[dataset_ids[idx2]]["sampled_tau_vals"][rng_ids[idx2],neuron_ids[idx2],:]))
            distance_matrix[idx2,idx1] = distance_matrix[idx1,idx2]
        end
    end
    return distance_matrix, deconvolved_activities, dataset_ids, rng_ids, neuron_ids, v_range, θh_range, P_range
end


"""
Finds the index in `neuron_ids` of a subset of neurons defined by their behavioral encoding, returns a dictionary with keys being the datasets.
Calling this function is necessary if you only wish to show a subset of neurons in your t-SNE plot.

# Arguments
- `neuron_ids_tsne`: List of neurons returned by `make_distance_matrix`.
- `neuron_categorization`: Dictionary of behavioral encoding returned by `categorize_all_neurons`
- `datasets`: Array of datasets to use.
- `rng`: Scalar indicating the time segment of interest.
- `beh_category`: String indicating the behavior of interest, e.g. "v" for velocity, "θh" for head curvature, "P" for pumping.
- `beh_subcategory` (optional, default is "all"): String indicating the behavioral subcategory of interest, e.g. "fwd" for forward neurons, "ventral" for ventral neurons.
"""
function find_subset_idx(neuron_ids_tsne, neuron_categorization, datasets, rng, beh_category; beh_subcategory="all")
    subset_idx_all = Dict()
    
    for dataset = datasets
        target = neuron_categorization[dataset][rng][beh_category][beh_subcategory]
        subset_idx = []

        if rng == 1
            idx_start = 0
        elseif rng == 2
            idx_start = length(neuron_categorization[dataset][rng-1]["all"])
        elseif rng == 3
            idx_start = length(neuron_categorization[dataset][rng-2]["all"]) + length(neuron_categorization[dataset][rng-1]["all"])
        elseif rng == 4
            idx_start = length(neuron_categorization[dataset][rng-3]["all"]) + length(neuron_categorization[dataset][rng-2]["all"]) + length(neuron_categorization[dataset][rng-1]["all"])
        end
        
        for i = 1:length(neuron_categorization[dataset][rng]["all"])
            idx = idx_start+i
            if issubset(neuron_ids_tsne[idx], target)
                push!(subset_idx, idx)
            end
        end
        
        subset_idx_all[dataset] = subset_idx
    end
    
    return subset_idx_all
end


"""
Runs t-SNE algorithm on `distance_matrix`, returns solution with lowest KL-divergence.

# Arguments
- `distance_matrix`: Distance matrix to run t-SNE on.
- `n_tsne`: Number of t-SNE attempts to run (per perplexity value)
- `perplexities`: Array of perplexity values to try
- `n_iters`: Number of iterations to run per t-SNE algorithm
- `subset` (optional, default is false): Whether you wish to only plot t-SNE with a subset of the neurons
- `subset_idx` (optional, default is an empty set): The list of neurons that you wish to plot t-SNE with for a particular dataset (i.e. subset_idx_all[dataset]), returned by `find_subset_idx`.
"""
function compute_tsne(distance_matrix, n_tsne, perplexities, n_iters; subset=false, subset_idx=[])
    all_kl_best = []
    all_tsne_best = []
    
    if subset
        distance_matrix = distance_matrix[subset_idx, subset_idx]
    else
        distance_matrix = distance_matrix
    end
    
    for perplexity in perplexities
        kl_best = Inf
        tsne_best = nothing
        for iter = 1:n_tsne
            tsne_dist, beta, kl = tsne(distance_matrix, 2, 0, n_iters, perplexity, verbose=false, distance=true, extended_output=true);
            if kl < kl_best
                kl_best = kl
                tsne_best = tsne_dist
            end
        end
        push!(all_kl_best, kl_best)
        push!(all_tsne_best, tsne_best)
    end
    return all_tsne_best, all_kl_best
end
