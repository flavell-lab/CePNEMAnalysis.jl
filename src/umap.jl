"""
    make_distance_matrix(
        datasets, fit_results, v_ranges, θh_ranges, P_ranges, neuron_categorization; 
        s_weight=2, v_weight=1, min_P_rng=[0,2], use_all_ranges=false, rngs_valid=nothing
    )

Makes a distance matrix (compatible with UMAP) of neurons between multiple datasets.

# Arguments:
- `datasets`: Array of datasets to use.
- `fit_results`: CePNEM fit results
- `v_ranges`: Range on velocity behavior
- `θh_ranges`: Range on head angle behavior
- `P_ranges`: Range on pumping behavior
- `neuron_categorization`: Categorization of neurons from `categorize_all_neurons` method
- `s_weight` (optional, default `1`): Weight of EWMA decay time `s`
- `v_weight` (optional, default `1`): Weight of velocity variable relative to other variables
- `min_P_rng` (optional, default `[0,2]`): Reject datasets and time ranges where pumping activity (25,75)th percentile doesn't include this range
- `use_all_ranges` (optional, default `false`): Use all avaialble ranges (vs select best range per dataset). 
    If `true`, could result in unequal number of ranges between datasets.
- `n_particles` (optional, default `2048`): Number of particles per neuron in the CePNEM fits
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
    find_subset_idx(neuron_ids_tsne, neuron_categorization, datasets, rng, beh_category; beh_subcategory="all")

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
    compute_tsne(distance_matrix, n_tsne, perplexities, n_iters; subset=false, subset_idx=[])

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


"""
    extrapolate_behaviors(fit_results, datasets, θh_pos_is_ventral) 

Creates extrapolated behaviors by appending behaviors from individual animals.

# Arguments
- `fit_results`: CePNEM fit results.
- `datasets`: Array of datasets to use.
- `θh_pos_is_ventral`: Whether positive θh value corresponds to ventral (`true`) or dorsal (`false`) head bending.
"""
function extrapolate_behaviors(fit_results, datasets, θh_pos_is_ventral) 
    all_behs = zeros(sum([length(fit_results[dataset]["v"]) for dataset in datasets]), 5)
    idx_beh = 1
    @showprogress for dataset in datasets
        len = length(fit_results[dataset]["v"])
        all_behs[idx_beh:idx_beh+len-1, 1] .= fit_results[dataset]["v"]
        all_behs[idx_beh:idx_beh+len-1, 2] .= fit_results[dataset]["θh"] .* (2*θh_pos_is_ventral[dataset]-1)
        all_behs[idx_beh:idx_beh+len-1, 3] .= fit_results[dataset]["P"]
        all_behs[idx_beh:idx_beh+len-1, 4] .= fit_results[dataset]["ang_vel"] .* (2*θh_pos_is_ventral[dataset]-1)
        all_behs[idx_beh:idx_beh+len-1, 5] .= fit_results[dataset]["curve"]
        idx_beh += len
    end
    return all_behs
end

"""
    compute_extrapolated_CePNEM_posterior_stats(
        fit_results, analysis_dict, datasets, θh_pos_is_ventral; n_idx=10001, use_pumping=true, normalize=true
    )

Computes statistics of the CePNEM fits of all neurons in each dataset across the set of extrapolated behaviors.

# Arguments
- `fit_results`: CePNEM fit results.
- `analysis_dict`: CePNEM fit analysis results dictionary.
- `datasets`: Array of datasets to use.
- `θh_pos_is_ventral`: Whether positive θh value corresponds to ventral (`true`) or dorsal (`false`) head bending.
- `n_idx` (optional, default `10001`): Number of particles in CePNEM fits.
- `use_pumping` (optional, default `true`): Whether to use pumping in CePNEM fits.
- `normalize` (optional, default `true`): Whether to normalize CePNEM fits by signal value.

# Returns
- `median_CePNEM_fits`: Median of the CePNEM fits of each neurons in each dataset across the set of extrapolated behaviors.
- `mean_CePNEM_fits`: Mean of the CePNEM fits of each neurons in each dataset across the set of extrapolated behaviors.
- `var_CePNEM_fits`: Variance of the CePNEM fits of each neurons in each dataset across the set of extrapolated behaviors.
"""
function compute_extrapolated_CePNEM_posterior_stats(fit_results, analysis_dict, datasets, θh_pos_is_ventral; n_idx=10001, use_pumping=true, normalize=true)
    median_CePNEM_fits = Dict()
    mean_CePNEM_fits = Dict()
    var_CePNEM_fits = Dict()
    all_behs = analysis_dict["extrapolated_behaviors"]
    @showprogress for dataset = datasets
        if dataset in keys(median_CePNEM_fits)
            continue
        end
        median_CePNEM_fits[dataset] = zeros(length(fit_results[dataset]["ranges"]), fit_results[dataset]["num_neurons"], size(all_behs,1))
        mean_CePNEM_fits[dataset] = zeros(length(fit_results[dataset]["ranges"]), fit_results[dataset]["num_neurons"], size(all_behs,1))
        var_CePNEM_fits[dataset] = zeros(length(fit_results[dataset]["ranges"]), fit_results[dataset]["num_neurons"], size(all_behs,1))
        for rng = 1:length(fit_results[dataset]["ranges"])
            for neuron = 1:fit_results[dataset]["num_neurons"]
                extrap = zeros(size(all_behs,1), n_idx)
                for idx = 1:n_idx
                    ps = deepcopy(fit_results[dataset]["sampled_trace_params"][rng,neuron,idx,1:8])
                    ps[3] = ps[3] * (2*θh_pos_is_ventral[dataset]-1)
                    ps[6] = 0
                    model = model_nl8(size(all_behs,1), ps..., all_behs[:,1], all_behs[:,2], (use_pumping ? all_behs[:,3] : zeros(length(all_behs[:,3]))))
                    if normalize
                        model = (model .- mean(model)) .* analysis_dict["signal"][dataset][neuron]
                    end
                    extrap[:,idx] .= model
                end
                median_CePNEM_fits[dataset][rng,neuron,:] .= median(extrap, dims=2)[:,1]
                mean_CePNEM_fits[dataset][rng,neuron,:] .= mean(extrap, dims=2)[:,1]
                var_CePNEM_fits[dataset][rng,neuron,:] .= var(extrap, dims=2)[:,1]
            end
        end
    end
    return median_CePNEM_fits, mean_CePNEM_fits, var_CePNEM_fits
end

"""
    append_median_CePNEM_fits(fit_results, analysis_dict, umap_dict, datasets)

Appends median CePNEM fits together into a single array.

# Arguments
- `fit_results`: CePNEM fit results.
- `analysis_dict`: CePNEM fit analysis results dictionary containing the `extrapolated_behaviors` key.
- `umap_dict`: UMAP results dictionary containing the `median_CePNEM_fits` key.
- `datasets`: Array of datasets to use.
"""
function append_median_CePNEM_fits(fit_results, analysis_dict, umap_dict, datasets)
    median_CePNEM_fits_all = zeros(2*sum([fit_results[d]["num_neurons"] for d in datasets]), size(analysis_dict["extrapolated_behaviors"], 1))
    count = 1
    for dataset in datasets
        if !(dataset in keys(umap_dict["median_CePNEM_fits"]))
            error("Dataset $(dataset) did not have median computed")
        end
        for rng=1:length(fit_results[dataset]["ranges"])
            for neuron=1:fit_results[dataset]["num_neurons"]
                median_CePNEM_fits_all[count,:] .= umap_dict["median_CePNEM_fits"][dataset][rng,neuron,:]
                count += 1
            end
        end
    end
    return median_CePNEM_fits_all
end

"""
    project_CePNEM_to_UMAP(fit_results, analysis_dict, umap_dict, datasets, θh_pos_is_ventral; n_idx=10001, use_pumping=true)

Projects median CePNEM fits to UMAP space.

# Arguments
- `fit_results`: CePNEM fit results.
- `analysis_dict`: CePNEM fit analysis results dictionary.
- `umap_dict`: UMAP results dictionary containing the `extrapolated_umap_median` key.
- `datasets`: Array of datasets to use.
- `θh_pos_is_ventral`: Whether positive θh value corresponds to ventral (`true`) or dorsal (`false`) head bending.
- `n_idx` (optional, default `10001`): Number of particles in CePNEM fits.
- `use_pumping` (optional, default `true`): Whether to use pumping in the model.
"""
function project_CePNEM_to_UMAP(fit_results, analysis_dict, umap_dict, datasets, θh_pos_is_ventral; n_idx=10001, use_pumping=true)
    umap_extrap_all_median = Dict()
    all_behs = analysis_dict["extrapolated_behaviors"]
    @showprogress for dataset = datasets
        umap_extrap_all_median[dataset] = Dict()
        for rng = 1:length(fit_results[dataset]["ranges"])
            umap_extrap_all_median[dataset][rng] = Dict()
            for neuron = 1:fit_results[dataset]["num_neurons"]
                extrap = zeros(size(all_behs,1), n_idx)
                for idx = 1:n_idx
                    ps = deepcopy(fit_results[dataset]["sampled_trace_params"][rng,neuron,idx,1:8])
                    ps[3] = ps[3] * (2*θh_pos_is_ventral[dataset]-1)
                    ps[6] = 0
                    model = model_nl8(size(all_behs,1), ps..., all_behs[:,1], all_behs[:,2], (use_pumping ? all_behs[:,3] : zeros(length(all_behs[:,3]))))
                    extrap[:,idx] .= (model .- mean(model)) .* analysis_dict["signal"][dataset][neuron]
                end
                umap_extrap_all_median[dataset][rng][neuron] = UMAP.transform(umap_dict["extrapolated_umap_median"], extrap)
            end
        end
    end
    return umap_extrap_all_median
end

"""
    make_umap_rgb(feature_imgs, feature_colors, full_umap_img, color_all, contrast)

Makes RGB image out of UMAP space.

# Arguments:
- `feature_imgs`: List of UMAP-projected images showing features of interest.
- `feature_colors`: List of UMAP-projected images showing colors of interest.
- `full_umap_img`: List of full UMAP image
- `color_all`: Background color of full UMAP image
- `contrast`: Contrast of features vs full UMAP image.
"""
function make_umap_rgb(feature_imgs, feature_colors, full_umap_img, color_all, contrast)
    log_hist_weights = reverse(transpose(log.(1 .+ full_umap_img)), dims=1)
    log_hist_weight_idx = zeros(length(feature_imgs), reverse(size(full_umap_img))...)
    for (i,img) in enumerate(feature_imgs)
        log_hist_weight_idx[i,:,:] .= reverse(transpose(log.(1 .+ img)), dims=1)
    end
    max_color = maximum(log_hist_weight_idx)
    img = zeros(size(log_hist_weights)...,3)
    img_all = contrast .* log_hist_weights ./ max_color
    for c=1:3
        img[:,:,c] .= sum([feature_colors[i][c] .* log_hist_weight_idx[i,:,:] ./ max_color for i=1:length(feature_imgs)])
    end

    img_color_sum = sum(img, dims=3)

    img_contrast = zeros(size(img))
    for c=1:3
        img_contrast[:,:,c] .= (img_color_sum .< img_all) .* img_all .* color_all[c] .+ (img_color_sum .>= img_all) .* img[:,:,c]
    end

    return img_contrast[:,:,1] .* RGB(1,0,0) .+ img_contrast[:,:,2] .* RGB(0,1,0) .+ img_contrast[:,:,3] .* RGB(0,0,1)
end


"""
    compute_umap_subcategories!(
        fit_results, analysis_dict, umap_dict, datasets; dataset_cats="2021-05-26-07", 
        rng_cats=1, ewma_step=5, ewma_max=50, suffix="_median", use_median=false
    )

Computes UMAP projections for each encoding category.

    # Arguments
    - `fit_results`: CePNEM fit results.
    - `analysis_dict`: CePNEM fit analysis results dictionary.
    - `umap_dict`: UMAP results dictionary. Modified in-place by this function.
    - `datasets`: Array of datasets to use.
    - `dataset_cats` (optional, default `"2021-05-26-07"`): Dataset to use for finding encoding categories.
    - `rng_cats` (optional, default `1`): Range to use for finding encoding categories.
    - `ewma_step` (optional, default `5`): Step size for exponentially weighted moving average.
    - `ewma_max` (optional, default `50`): Maximum number of steps for exponentially weighted moving average.
    - `suffix` (optional, default `"_median"`): Suffix for extrapolated UMAP key in `analysis_dict`.
    - `use_median` (optional, default `false`): Whether to use median (`true`) or all posterior points (`false`) for extrapolated UMAP projections.
"""
function compute_umap_subcategories!(fit_results, analysis_dict, umap_dict, datasets; dataset_cats="2021-05-26-07", rng_cats=1, ewma_step=5, ewma_max=50, suffix="_median", use_median=false)
    xmin = umap_dict["umap_xmin"]
    xmax = umap_dict["umap_xmax"]
    xstep = umap_dict["umap_xstep"]
    
    ymin = umap_dict["umap_ymin"]
    ymax = umap_dict["umap_ymax"]
    ystep = umap_dict["umap_ystep"]
    xaxis = umap_dict["umap_xaxis"]
    yaxis = umap_dict["umap_yaxis"]
    
    hist_weights = zeros(length(xaxis)-1, length(yaxis)-1)
    
    hist_weights_cats = Dict()
    hist_weights_subcats = Dict()
        
    hist_weights_cats_ec = Dict()
        
    
    for k_beh in keys(analysis_dict["neuron_categorization"][dataset_cats][rng_cats])
        hist_weights_cats[k_beh] = Dict()
        if k_beh == "all"
            hist_weights_cats[k_beh] = zeros(length(xaxis)-1, length(yaxis)-1)
            continue
        end
        for k_subcat in keys(analysis_dict["neuron_categorization"][dataset_cats][rng_cats][k_beh])
            hist_weights_cats[k_beh][k_subcat] = zeros(length(xaxis)-1, length(yaxis)-1)
        end
        hist_weights_subcats[k_beh] = Dict()
        for k_subcat in keys(analysis_dict["neuron_subcategorization"][dataset_cats][rng_cats][k_beh])
            hist_weights_subcats[k_beh][k_subcat] = zeros(length(xaxis)-1, length(yaxis)-1)
        end
    end
    
    ewma_vals = 0:ewma_step:ewma_max
    hist_weights_ewma = zeros(length(ewma_vals), length(xaxis)-1, length(yaxis)-1)
        
    @showprogress for dataset = datasets
        for rng = 1:length(fit_results[dataset]["ranges"])
            for neuron = 1:fit_results[dataset]["num_neurons"]
                if use_median
                    hist_fit = fit(Histogram, ([median(umap_dict["umap_extrap_all$(suffix)"][dataset][rng][neuron][1,:])], [median(umap_dict["umap_extrap_all$(suffix)"][dataset][rng][neuron][2,:])]), (xaxis, yaxis))
                else
                    hist_fit = fit(Histogram, (umap_dict["umap_extrap_all$(suffix)"][dataset][rng][neuron][1,:], umap_dict["umap_extrap_all$(suffix)"][dataset][rng][neuron][2,:]), (xaxis, yaxis))
                end
                
                for k_beh in keys(analysis_dict["neuron_categorization"][dataset][rng])
                    if k_beh == "all"
                        if neuron in analysis_dict["neuron_categorization"][dataset][rng][k_beh]
                            hist_weights_cats[k_beh] .+= hist_fit.weights
                        end
                        continue
                    end
                    for k_cat in keys(analysis_dict["neuron_categorization"][dataset][rng][k_beh])
                        if neuron in analysis_dict["neuron_categorization"][dataset][rng][k_beh][k_cat]
                            hist_weights_cats[k_beh][k_cat] .+= hist_fit.weights
                        end
                    end

                    for k_subcat in keys(analysis_dict["neuron_subcategorization"][dataset][rng][k_beh])
                        if neuron in analysis_dict["neuron_subcategorization"][dataset][rng][k_beh][k_subcat]
                            hist_weights_subcats[k_beh][k_subcat] .+= hist_fit.weights
                        end
                    end
                end

                for idx = 1:size(umap_dict["umap_extrap_all$(suffix)"][dataset][rng][neuron],2)
                    loc_x = Int((umap_dict["umap_extrap_all$(suffix)"][dataset][rng][neuron][1,idx] - xmin) ÷ xstep + 1)
                    loc_y = Int((umap_dict["umap_extrap_all$(suffix)"][dataset][rng][neuron][2,idx] - ymin) ÷ ystep + 1)
                    ewma_val = Int(min(ewma_max, fit_results[dataset]["sampled_tau_vals"][rng, neuron, idx]) ÷ ewma_step) + 1
                    hist_weights_ewma[ewma_val, loc_x, loc_y] += 1
                end 

                hist_weights .+= hist_fit.weights
            end
        end    
    end
    
    umap_dict["umap_hist_weights$(suffix)"] = hist_weights
    umap_dict["umap_hist_weights_cats$(suffix)"] = hist_weights_cats
    umap_dict["umap_hist_weights_subcats$(suffix)"] = hist_weights_subcats
    umap_dict["umap_hist_weights_ewma$(suffix)"] = hist_weights_ewma;

    return nothing
end
