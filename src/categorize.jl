"""
Evaluate model NL7b, deconvolved, given `params` and `v`, `θh`, and `P`. Does not use the sigmoid.
"""
function deconvolved_model_nl7b(params::Vector{Float64}, v::Float64, θh::Float64, P::Float64)
    return ((params[1]+1)/sqrt(params[1]^2+1) - 2*params[1]/sqrt(params[1]^2+1)*(v/v_STD < 0)) * (
        params[2] * (v/v_STD) + params[3] * (θh/θh_STD) + params[4] * (P/P_STD) + params[5]) + params[8]
end

"""
Computes the valid range of a behavior `beh` (eg: velocity cropped to a given time range).
Computes percentile based on `thresh`, and uses 4 points instead of 2 for velocity (`beh_idx = 1`)
"""
function compute_range(beh::Vector{Float64}, thresh::Real, beh_idx::Int)
    @assert(thresh < 50)
    if beh_idx == 1
        min_beh = percentile(beh[beh .< 0], 2*thresh)
        max_beh = percentile(beh[beh .> 0], 100-2*thresh)
    else
        min_beh = percentile(beh, thresh)
        max_beh = percentile(beh, 100-thresh)
    end
    if beh_idx != 3
        @assert(min_beh < 0)
        @assert(max_beh > 0)
    end
    
    if beh_idx == 1
        return [min_beh, min_beh/100, max_beh/100, max_beh]
    else
        return [min_beh, max_beh]
    end
end

"""
Computes deconvolved activity of model NL7b for each sampled parameter in `sampled_trace_params`,
at a lattice defined by `v_rng`, `θh_rng`, and `P_rng`.
"""
function get_deconvolved_activity(sampled_trace_params, v_rng, θh_rng, P_rng)
    n_traces = size(sampled_trace_params,1)
    deconvolved_activity = zeros(n_traces, length(v_rng), length(θh_rng), length(P_rng))
    for x in 1:n_traces
        for (i,v_) in enumerate(v_rng)
            for (j,θh_) in enumerate(θh_rng)
                for (k,P_) in enumerate(P_rng)
                    deconvolved_activity[x,i,j,k] = deconvolved_model_nl7b(sampled_trace_params[x,:],v_,θh_,P_)
                end
            end
        end
    end

    return deconvolved_activity
end


"""
Makes deconvolved lattices for each dataset, time range, and neuron in `fit_results`.
Returns velocity, head angle, and pumping ranges, and the deconvolved activity of each neuron at each lattice point defined by them,
for both statistically useful ranges (first return value), and full ranges (second return value) designed for plotting consistency.
"""
function make_deconvolved_lattice(fit_results)
    deconvolved_activity = Dict()
    v_ranges = Dict()
    θh_ranges = Dict()
    P_ranges = Dict()
    deconvolved_activity_plot = Dict()
    v_ranges_plot = Dict()
    θh_ranges_plot = Dict()
    P_ranges_plot = Dict()
    @showprogress for dataset in keys(fit_results)
        deconvolved_activity[dataset] = Dict()
        v_ranges[dataset] = Dict()
        θh_ranges[dataset] = Dict()
        P_ranges[dataset] = Dict()

        deconvolved_activity_plot[dataset] = Dict()

        v_all = fit_results[dataset]["v"]
        θh_all = fit_results[dataset]["θh"]
        P_all = fit_results[dataset]["P"]
        
        v_ranges_plot[dataset] = compute_range(v_all, 5, 1)
        θh_ranges_plot[dataset] = compute_range(θh_all, 5, 2)
        P_ranges_plot[dataset] = compute_range(P_all, 5, 3)

        for rng=1:4
            deconvolved_activity[dataset][rng] = Dict()

            deconvolved_activity_plot[dataset][rng] = Dict()

            v = v_all[fit_results[dataset]["ranges"][rng]]
            θh = θh_all[fit_results[dataset]["ranges"][rng]]
            P = P_all[fit_results[dataset]["ranges"][rng]]

            results = fit_results[dataset]["sampled_trace_params"]

            v_ranges[dataset][rng] = compute_range(v, 25, 1)
            θh_ranges[dataset][rng] = compute_range(θh, 25, 2)
            P_ranges[dataset][rng] = compute_range(P, 25, 3)
            
            

            for neuron=1:size(results,2)
                deconvolved_activity[dataset][rng][neuron] =
                        get_deconvolved_activity(results[rng,neuron,:,:], v_ranges[dataset][rng],
                                θh_ranges[dataset][rng], P_ranges[dataset][rng])
                
                deconvolved_activity_plot[dataset][rng][neuron] =
                        get_deconvolved_activity(results[rng,neuron,:,:], v_ranges_plot[dataset],
                                θh_ranges_plot[dataset], P_ranges_plot[dataset])
            end
        end
    end
    return (v_ranges, θh_ranges, P_ranges, deconvolved_activity), (v_ranges_plot, θh_ranges_plot, P_ranges_plot, deconvolved_activity_plot)
end


"""
Computes neuron p-values by computing differences between two different deconvolved activities.
To find encoding of a neuron, set the second activity to 0.
To find encoding change, set it to a different time window.
To find distance between neurons, set `compute_p = false` and specify the `metric` (default `abs`)
to use to compare medians of the two posteriors.
"""
function neuron_p_vals(deconvolved_activity_1, deconvolved_activity_2; compute_p::Bool=true, metric::Function=abs)
    categories = Dict()
    
    s = size(deconvolved_activity_1)
    categories["v_encoding"] = compute_p ? ones(s[2], s[2], s[3], s[4]) : zeros(s[2], s[2], s[3], s[4])
    
    for i in 1:s[2]-1
        for j in i+1:s[2]
            for k in 1:s[3]
                for m in 1:s[4]
                    # count equal points as 0.5
                    diff_1 = deconvolved_activity_1[:,i,k,m] .- deconvolved_activity_1[:,j,k,m]
                    diff_2 = deconvolved_activity_2[:,i,k,m] .- deconvolved_activity_2[:,j,k,m]
                    categories["v_encoding"][i,j,k,m] = compute_p ? prob_P_greater_Q(diff_1, diff_2) : metric(median(diff_1) - median(diff_2))
                    categories["v_encoding"][j,i,k,m] = compute_p ? 1-categories["v_encoding"][i,j,k,m] : categories["v_encoding"][i,j,k,m]
                end
            end
        end
    end
            
    for i = [1,4]
        k = (i == 1) ? "rev_θh_encoding" : "fwd_θh_encoding"
        diff_1 = deconvolved_activity_1[:,i,1,:] .- deconvolved_activity_1[:,i,2,:]
        diff_2 = deconvolved_activity_2[:,i,1,:] .- deconvolved_activity_2[:,i,2,:]
        categories[k] = compute_p ? prob_P_greater_Q(diff_1, diff_2) : metric(median(diff_1) - median(diff_2))

        k = (i == 1) ? "rev_P_encoding" : "fwd_P_encoding"
        diff_1 = deconvolved_activity_1[:,i,:,1] .- deconvolved_activity_1[:,i,:,2]
        diff_2 = deconvolved_activity_2[:,i,:,1] .- deconvolved_activity_2[:,i,:,2]
        categories[k] = compute_p ? prob_P_greater_Q(diff_1, diff_2) : metric(median(diff_1) - median(diff_2))
    end
    
    return categories
end

"""
Categorizes all neurons from their deconvolved activities.

# Arguments:
- `deconvolved_activities_1`: Deconvolved activities of neurons.
- `deconvolved_activities_2`: Either 0 (to check neuron encoding), or deconvolved activities of neurons at a different time point (to check encoding change).
- `p`: Significant p-value.
- `θh_pos_is_ventral`: Whether positive θh value corresponds to ventral (`true`) or dorsal (`false`) head bending.
"""
function categorize_neurons(deconvolved_activities_1, deconvolved_activities_2, p::Real, θh_pos_is_ventral::Bool)
    categories = Dict()
    categories["v"] = Dict()
    categories["v"]["rev"] = []
    categories["v"]["fwd"] = []
    categories["v"]["rev_slope_pos"] = []
    categories["v"]["rev_slope_neg"] = []
    categories["v"]["rect_pos"] = []
    categories["v"]["rect_neg"] = []
    categories["v"]["fwd_slope_pos"] = []
    categories["v"]["fwd_slope_neg"] = []
    categories["v"]["all"] = []
    categories["θh"] = Dict()
    categories["θh"]["rect_fwd_ventral"] = []
    categories["θh"]["rect_fwd_dorsal"] = []
    categories["θh"]["rect_rev_ventral"] = []
    categories["θh"]["rect_rev_dorsal"] = []
    categories["θh"]["all"] = []
    categories["P"] = Dict()
    categories["P"]["rect_fwd_act"] = []
    categories["P"]["rect_fwd_inh"] = []
    categories["P"]["rect_rev_act"] = []
    categories["P"]["rect_rev_inh"] = []
    categories["P"]["all"] = []

    neuron_cats = Dict()
    for neuron = keys(deconvolved_activities_1)
        neuron_cats[neuron] = neuron_p_vals(deconvolved_activities_1[neuron], deconvolved_activities_2[neuron])
    end
    
    max_n = maximum(keys(neuron_cats))

    corrected_p_vals = Dict()
    corrected_p_vals["v"] = Dict()
    corrected_p_vals["v"]["rev"] = ones(max_n)
    corrected_p_vals["v"]["fwd"] = ones(max_n)
    corrected_p_vals["v"]["rev_slope_pos"] = ones(max_n)
    corrected_p_vals["v"]["rev_slope_neg"] = ones(max_n)
    corrected_p_vals["v"]["rect_pos"] = ones(max_n)
    corrected_p_vals["v"]["rect_neg"] = ones(max_n)
    corrected_p_vals["v"]["fwd_slope_pos"] = ones(max_n)
    corrected_p_vals["v"]["fwd_slope_neg"] = ones(max_n)
    corrected_p_vals["v"]["all"] = ones(max_n)
    corrected_p_vals["θh"] = Dict()
    corrected_p_vals["θh"]["rect_fwd_ventral"] = ones(max_n)
    corrected_p_vals["θh"]["rect_fwd_dorsal"] = ones(max_n)
    corrected_p_vals["θh"]["rect_rev_ventral"] = ones(max_n)
    corrected_p_vals["θh"]["rect_rev_dorsal"] = ones(max_n)
    corrected_p_vals["θh"]["all"] = ones(max_n)
    corrected_p_vals["P"] = Dict()
    corrected_p_vals["P"]["rect_fwd_act"] = ones(max_n)
    corrected_p_vals["P"]["rect_fwd_inh"] = ones(max_n)
    corrected_p_vals["P"]["rect_rev_act"] = ones(max_n)
    corrected_p_vals["P"]["rect_rev_inh"] = ones(max_n)
    corrected_p_vals["P"]["all"] = ones(max_n)
    
    v_p_vals_uncorr = ones(max_n,4,4)
    v_p_vals = ones(max_n,4,4)
    
    v_p_vals_all_uncorr = ones(max_n)
    p_vals_all_uncorr = ones(max_n)
    v_p_vals_all = ones(max_n)
    p_vals_all = ones(max_n)
    
    # for velocity, take best θh and P values but MH correct
    for i=1:4
        for j=1:4
            for neuron in keys(neuron_cats)
                v_p_vals_uncorr[neuron,i,j] = minimum(adjust(reshape(neuron_cats[neuron]["v_encoding"][i,j,:,:], (4,)), BenjaminiHochberg()))
            end
            # use BH correction since these are independent values
            v_p_vals[:,i,j] .= adjust(v_p_vals_uncorr[:,i,j], BenjaminiHochberg())
        end
    end

    for neuron in keys(neuron_cats)
        adjust_v_p_vals = Vector{Float64}()
        all_p_vals = Vector{Float64}()
        for x in 1:3
            for y=x+1:4
                # correct for anticorrelation between forward and reverse neurons
                push!(adjust_v_p_vals, min(1,2*min(v_p_vals_uncorr[neuron,x,y], v_p_vals_uncorr[neuron,y,x])))
                push!(all_p_vals, min(1,2*min(v_p_vals_uncorr[neuron,x,y], v_p_vals_uncorr[neuron,y,x])))
            end
        end
        n = neuron
        push!(all_p_vals, min(1,4*min(neuron_cats[n]["fwd_θh_encoding"], 1 - neuron_cats[n]["fwd_θh_encoding"], 
                    neuron_cats[n]["rev_θh_encoding"], 1 - neuron_cats[n]["rev_θh_encoding"])))
        push!(all_p_vals, min(1,4*min(neuron_cats[n]["fwd_P_encoding"], 1 - neuron_cats[n]["fwd_P_encoding"],
                    neuron_cats[n]["rev_P_encoding"], 1 - neuron_cats[n]["rev_P_encoding"])))
        # use BH correction since correlations are expected to be positive after above correction
        p_vals_all_uncorr[neuron] = minimum(adjust(all_p_vals, BenjaminiHochberg()))
        v_p_vals_all_uncorr[neuron] = minimum(adjust(adjust_v_p_vals, BenjaminiHochberg()))
    end
    
    v_p_vals_all = adjust(v_p_vals_all_uncorr, BenjaminiHochberg())
    p_vals_all = adjust(p_vals_all_uncorr, BenjaminiHochberg())
    
    
    categories["v"]["rev"] = [n for n in 1:max_n if v_p_vals[n,4,1] < p]
    categories["v"]["fwd"] = [n for n in 1:max_n if v_p_vals[n,1,4] < p]
    categories["v"]["rev_slope_pos"] = [n for n in 1:max_n if v_p_vals[n,1,2] < p]
    categories["v"]["rev_slope_neg"] = [n for n in 1:max_n if v_p_vals[n,2,1] < p]
    categories["v"]["rect_pos"] = [n for n in 1:max_n if v_p_vals[n,2,3] < p]
    categories["v"]["rect_neg"] = [n for n in 1:max_n if v_p_vals[n,3,2] < p]
    categories["v"]["fwd_slope_pos"] = [n for n in 1:max_n if v_p_vals[n,3,4] < p]
    categories["v"]["fwd_slope_neg"] = [n for n in 1:max_n if v_p_vals[n,4,3] < p]
    categories["v"]["all"] = [n for n in 1:max_n if v_p_vals_all[n] < p]

    corrected_p_vals["v"]["rev"] .= v_p_vals[n,4,1]
    corrected_p_vals["v"]["fwd"] .= v_p_vals[n,1,4]
    corrected_p_vals["v"]["rev_slope_pos"] .= v_p_vals[n,1,2]
    corrected_p_vals["v"]["rev_slope_neg"] .= v_p_vals[n,2,1]
    corrected_p_vals["v"]["rect_pos"] .= v_p_vals[n,2,3]
    corrected_p_vals["v"]["rect_neg"] .= v_p_vals[n,3,2]
    corrected_p_vals["v"]["fwd_slope_pos"] .= v_p_vals[n,3,4]
    corrected_p_vals["v"]["fwd_slope_neg"] .= v_p_vals[n,4,3]
    corrected_p_vals["v"]["all"] .= v_p_vals_all[n]
    
    if !θh_pos_is_ventral
        fwd_θh_dorsal = adjust([neuron_cats[n]["fwd_θh_encoding"] for n in 1:max_n], BenjaminiHochberg())
        fwd_θh_ventral = adjust([1 - neuron_cats[n]["fwd_θh_encoding"] for n in 1:max_n], BenjaminiHochberg())
        rev_θh_dorsal = adjust([neuron_cats[n]["rev_θh_encoding"] for n in 1:max_n], BenjaminiHochberg())
        rev_θh_ventral = adjust([1 - neuron_cats[n]["rev_θh_encoding"] for n in 1:max_n], BenjaminiHochberg())
        θh_ventral = adjust([min(1 - neuron_cats[n]["fwd_θh_encoding"], 1 - neuron_cats[n]["rev_θh_encoding"]) for n in 1:max_n], BenjaminiHochberg())
        θh_dorsal = adjust([min(neuron_cats[n]["fwd_θh_encoding"], neuron_cats[n]["rev_θh_encoding"]) for n in 1:max_n], BenjaminiHochberg())
    else
        fwd_θh_dorsal = adjust([1 - neuron_cats[n]["fwd_θh_encoding"] for n in 1:max_n], BenjaminiHochberg())
        fwd_θh_ventral = adjust([neuron_cats[n]["fwd_θh_encoding"] for n in 1:max_n], BenjaminiHochberg())
        rev_θh_dorsal = adjust([1 - neuron_cats[n]["rev_θh_encoding"] for n in 1:max_n], BenjaminiHochberg())
        rev_θh_ventral = adjust([neuron_cats[n]["rev_θh_encoding"] for n in 1:max_n], BenjaminiHochberg())
        θh_dorsal = adjust([min(1 - neuron_cats[n]["fwd_θh_encoding"], 1 - neuron_cats[n]["rev_θh_encoding"]) for n in 1:max_n], BenjaminiHochberg())
        θh_ventral = adjust([min(neuron_cats[n]["fwd_θh_encoding"], neuron_cats[n]["rev_θh_encoding"]) for n in 1:max_n], BenjaminiHochberg())
    end
    θh_all = adjust([min(neuron_cats[n]["fwd_θh_encoding"], 1 - neuron_cats[n]["fwd_θh_encoding"],
            neuron_cats[n]["rev_θh_encoding"], 1 - neuron_cats[n]["rev_θh_encoding"]) for n in 1:max_n], BenjaminiHochberg())
    
    categories["θh"]["rect_fwd_ventral"] = [n for n in 1:max_n if fwd_θh_ventral[n] < p]
    categories["θh"]["rect_fwd_dorsal"] = [n for n in 1:max_n if fwd_θh_dorsal[n] < p]
    categories["θh"]["rect_rev_ventral"] = [n for n in 1:max_n if rev_θh_ventral[n] < p]
    categories["θh"]["rect_rev_dorsal"] = [n for n in 1:max_n if rev_θh_dorsal[n] < p]
    # use Bonferroni correction since these are expected to be anticorrelated
    categories["θh"]["dorsal"] = [n for n in 1:max_n if θh_dorsal[n] < p/2]
    categories["θh"]["ventral"] = [n for n in 1:max_n if θh_ventral[n] < p/2]
    categories["θh"]["all"] = [n for n in 1:max_n if θh_all[n] < p/4]

    corrected_p_vals["θh"]["rect_fwd_ventral"] .= fwd_θh_ventral
    corrected_p_vals["θh"]["rect_fwd_dorsal"] .= fwd_θh_dorsal
    corrected_p_vals["θh"]["rect_rev_ventral"] .= rev_θh_ventral
    corrected_p_vals["θh"]["rect_rev_dorsal"] .= rev_θh_dorsal
    # use Bonferroni correction since these are expected to be anticorrelated
    corrected_p_vals["θh"]["dorsal"] .= θh_dorsal .* 2
    corrected_p_vals["θh"]["ventral"] .= θh_ventral .* 2
    corrected_p_vals["θh"]["all"] .= θh_all .* 4
    
    
    fwd_P_act = adjust([neuron_cats[n]["fwd_P_encoding"] for n in 1:max_n], BenjaminiHochberg())
    fwd_P_inh = adjust([1 - neuron_cats[n]["fwd_P_encoding"] for n in 1:max_n], BenjaminiHochberg())
    rev_P_act = adjust([neuron_cats[n]["rev_P_encoding"] for n in 1:max_n], BenjaminiHochberg())
    rev_P_inh = adjust([1 - neuron_cats[n]["rev_P_encoding"] for n in 1:max_n], BenjaminiHochberg())
    P_all = adjust([min(neuron_cats[n]["fwd_P_encoding"], 1 - neuron_cats[n]["fwd_P_encoding"],
            neuron_cats[n]["rev_P_encoding"], 1 - neuron_cats[n]["rev_P_encoding"]) for n in 1:max_n], BenjaminiHochberg())
    P_act = adjust([min(neuron_cats[n]["fwd_P_encoding"], neuron_cats[n]["rev_P_encoding"]) for n in 1:max_n], BenjaminiHochberg())
    P_inh = adjust([min(1 - neuron_cats[n]["fwd_P_encoding"], 1 - neuron_cats[n]["rev_P_encoding"]) for n in 1:max_n], BenjaminiHochberg())
    
    categories["P"]["rect_fwd_inh"] = [n for n in 1:max_n if fwd_P_inh[n] < p]
    categories["P"]["rect_fwd_act"] = [n for n in 1:max_n if fwd_P_act[n] < p]
    categories["P"]["rect_rev_inh"] = [n for n in 1:max_n if rev_P_inh[n] < p]
    categories["P"]["rect_rev_act"] = [n for n in 1:max_n if rev_P_act[n] < p]
    # use Bonferroni correction since these are expected to be anticorrelated
    categories["P"]["act"] = [n for n in 1:max_n if P_act[n] < p/2]
    categories["P"]["inh"] = [n for n in 1:max_n if P_inh[n] < p/2]
    categories["P"]["all"] = [n for n in 1:max_n if P_all[n] < p/4]
    
    categories["all"] = [n for n in 1:max_n if p_vals_all[n] < p]

    corrected_p_vals["P"]["rect_fwd_inh"] .= fwd_P_inh
    corrected_p_vals["P"]["rect_fwd_act"] .= fwd_P_act
    corrected_p_vals["P"]["rect_rev_inh"] .= rev_P_inh
    corrected_p_vals["P"]["rect_rev_act"] .= rev_P_act
    # use Bonferroni correction since these are expected to be anticorrelated
    corrected_p_vals["P"]["act"] .= P_act .* 2
    corrected_p_vals["P"]["inh"] .= P_inh .* 2
    corrected_p_vals["P"]["all"] .= P_all .* 4
    
    corrected_p_vals["all"] .= p_vals_all
    
    return categories, corrected_p_vals
end


