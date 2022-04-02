function make_deconvolved_heatmap(deconvolved_activity, axis; res=200)
    @assert(res % 2 == 0)
    all_hmap = zeros(size(deconvolved_activity,1),res, res)
    for i=1:size(deconvolved_activity,1)
        if axis == 2
            pts1 = dropdims(mean(deconvolved_activity[i,:,1,:], dims=2), dims=2)
            pts2 = dropdims(mean(deconvolved_activity[i,:,2,:], dims=2), dims=2)
        elseif axis == 3
            pts1 = dropdims(mean(deconvolved_activity[i,:,:,1], dims=2), dims=2)
            pts2 = dropdims(mean(deconvolved_activity[i,:,:,2], dims=2), dims=2)
        else
            error("Axis must be 2 (θh) or 3 (pumping)")
        end
        s = Int(res/2)
        for j=1:s
            all_hmap[i,j,1] = (pts1[1] * (s-j)/(s-1) + pts1[2] * (j-1)/(s-1))
            all_hmap[i,j,end] = (pts2[1] * (s-j)/(s-1) + pts2[2] * (j-1)/(s-1))
        end
        for j=1:s
            all_hmap[i,j+s,1] = (pts1[3] * (s-j)/(s-1) + pts1[4] * (j-1)/(s-1))
            all_hmap[i,j+s,end] = (pts2[3] * (s-j)/(s-1) + pts2[4] * (j-1)/(s-1))
        end
        for j=1:res
            for k=2:res-1
                all_hmap[i,j,k] = (all_hmap[i,j,1] * (res-k)/(res-1) + all_hmap[i,j,end] * (k-1)/(res-1))
            end
        end
    end
    return all_hmap
end

"""
Plots a t-SNE embedding between datasets, ranges, and neurons.

# Arguments
- `tsne_dist`: t-SNE embedding matrix
- `dataset_ids_tsne`: Dataset of each row in `tsne_dist`
- `range_ids_tsne`: Range of each row in `tsne_dist`
- `neuron_ids_tsne`: Neuron of each row in `tsne_dist`
- `neuron_categorization`: Categorization of each neuron
- `vars_plot`: List of which variables to plot colors of.
    1 = velocity, 2 = head angle, 3 = pumping. If empty, plots datasets and ranges in different colors. If `[4]`, plot EWMA
- `label_v_shapes` (optional, default `false`): If set, plot different velocity neurons with different shapes.
- `label_neurons` (optional, default `false`): If set, label each point in the plot with the neuron ID.
- `plot_axes` (optional, default `false`): If set, plot the numerical values of the axes.
- `velocity_colors` (optional, default `[RGB.(0,0.9,1), RGB.(0.5,0.6,1), RGB.(0,0,1)]`): The colors of forward, unknown/other velocity, and reversal neurons.
- `θh_colors` (optional, default `[RGB.(1,0.7,0.3), RGB.(1,0.3,0.7), RGB.(0.7,0,0)]`): The colors of ventral, unknown/other head angle, and dorsal neurons.
- `P_colors` (optional, default `[RGB.(0,1,0), RGB.(0.7,1,0), RGB.(0,0.5,0)]`): The colors of activated, unknown/other pumping, and inhibited neurons.
- `c_multiplex` (optional, default `RGB.(0.7,0.7,0.7)`): The colors of multiplexed neurons.
- `c_nonencoding` (optional, default `RGB.(0.8,0.8,0.8)`): The colors of non-encoding neurons.
- `shapes` (optional, default `[:circle, :rtriangle, :ltriangle, :ldiamond]`): The shapes of non-velocity, forward, reversal, and unknown/other velocity neurons.
    If `label_v_shapes` is false, only the first element matters.
- `sizes` (optiona, default `[9,14,14,12]`): The marker sizes. If `label_v_shapes` is false, only the first element matters.
- `s_rng`: Range of `s` values to consider.
- `s_res`: Resolution of different `s` values.
"""
function plot_tsne(tsne_dist, fit_results, dataset_ids_tsne, range_ids_tsne, neuron_ids_tsne, neuron_categorization, vars_plot; label_v_shapes=false, label_neurons=false, plot_axes=false,
        velocity_colors=[RGB.(0,0.9,1), RGB.(0.5,0.6,1), RGB.(0,0,1)], θh_colors=[RGB.(1,0.7,0.3), RGB.(1,0.3,0.7), RGB.(0.7,0,0)], P_colors=[RGB.(0,1,0), RGB.(0.7,1,0), RGB.(0,0.5,0)], 
        c_multiplex=RGB.(0.7,0.7,0.7), c_nonencoding=RGB.(0.8,0.8,0.8), shapes=[:circle, :rtriangle, :ltriangle, :diamond], sizes=[6,10,10,8], s_rng=[0,7], s_res=100)
    if label_v_shapes
        @assert(!isnothing(neuron_categorization), "Neuron categories must exist to label velocity with shapes.")
    end
    if length(vars_plot) > 0 && !(4 in vars_plot)
        colors = [velocity_colors, θh_colors, P_colors]
    end

    Plots.plot()
    color_idx = 0
    dataset_prev = nothing
    rng_prev = -1
    for i=1:size(tsne_dist,1)
        dataset = dataset_ids_tsne[i]
        rng = range_ids_tsne[i]
        if dataset != dataset_prev || rng != rng_prev
            dataset_prev = dataset
            rng_prev = rng
            color_idx += 1
        end
        n = neuron_ids_tsne[i]
        is_v = n in neuron_categorization[dataset][rng]["v"]["all"]
        is_θh = n in neuron_categorization[dataset][rng]["θh"]["all"]
        is_P = n in neuron_categorization[dataset][rng]["P"]["all"]

        if length(vars_plot) > 0 && !(4 in vars_plot)
            c = nothing
            is_multiplex = (is_v && 1 in vars_plot) + (is_θh && 2 in vars_plot) + (is_P && 3 in vars_plot) > 1

            if is_multiplex
                c = c_multiplex
            elseif is_v && 1 in vars_plot
                is_fwd = n in neuron_categorization[dataset][rng]["v"]["fwd"]
                is_rev = n in neuron_categorization[dataset][rng]["v"]["rev"]
                if is_fwd
                    c = colors[1][1]
                elseif is_rev
                    c = colors[1][3]
                else
                    c = colors[1][2]
                end
            elseif is_θh && 2 in vars_plot
                num_enc = 0
                if n in neuron_categorization[dataset][rng]["θh"]["rect_fwd_dorsal"] ||
                        n in neuron_categorization[dataset][rng]["θh"]["rect_rev_dorsal"]
                    c = colors[2][3]
                    num_enc += 1
                end
                if n in neuron_categorization[dataset][rng]["θh"]["rect_fwd_ventral"] ||
                    n in neuron_categorization[dataset][rng]["θh"]["rect_rev_ventral"]
                    c = colors[2][1]
                    num_enc += 1
                end
                if num_enc != 1
                    c = colors[2][2]
                end
            elseif is_P && 3 in vars_plot
                num_enc = 0
                if n in neuron_categorization[dataset][rng]["P"]["rect_fwd_inh"] || 
                        n in neuron_categorization[dataset][rng]["P"]["rect_rev_inh"]
                    c = colors[3][3]
                    num_enc += 1
                end
                if n in neuron_categorization[dataset][rng]["P"]["rect_fwd_act"] || 
                    n in neuron_categorization[dataset][rng]["P"]["rect_rev_act"]
                    c = colors[3][1]
                    num_enc += 1
                end
                if num_enc != 1
                    c = colors[3][2]
                end
            else
                c = c_nonencoding
            end
        elseif 4 in vars_plot
            s = median(fit_results[dataset]["sampled_tau_vals"][rng,n,:])
            s_idx = min(s_res, Int(round(s_res*(s-s_rng[1]) / (s_rng[end]-s_rng[1]))))
            c = palette(:thermal, s_res+1)[s_idx+1]
        else
            c = palette(:default)[color_idx]
        end

        if label_v_shapes
            is_fwd = n in neuron_categorization[dataset][rng]["v"]["fwd"]
            is_rev = n in neuron_categorization[dataset][rng]["v"]["rev"]
            if !is_v
                shape_idx = 1
            elseif is_fwd
                shape_idx = 2
            elseif is_rev
                shape_idx = 3
            else
                shape_idx = 4
            end
        else
            shape_idx = 1
        end
        
        if label_neurons
            Plots.scatter!([tsne_dist[i,1]], [tsne_dist[i,2]], label=nothing, color=c, markerstrokecolor=c, markersize=sizes[shape_idx], markershape=shapes[shape_idx], series_annotations=Plots.text.([n], :bottom, 7))
        else
            Plots.scatter!([tsne_dist[i,1]], [tsne_dist[i,2]], label=nothing, color=c, markerstrokecolor=c, markersize=sizes[shape_idx], markershape=shapes[shape_idx])
        end
    end
    if !plot_axes
        Plots.plot!(xaxis=nothing, yaxis=nothing)
    end
    Plots.plot!(framestyle=:box, size=(600,600))
end

"""
Plots histogram of tau (half-decay) times for all encoding neurons.
"""
function plot_tau_histogram(fit_results, neuron_categorization)
    s_vals = []
    for dataset in keys(fit_results)
        for rng in 1:length(fit_results[dataset]["ranges"])
            append!(s_vals, dropdims(median(fit_results[dataset]["sampled_tau_vals"][rng,neuron_categorization[dataset][rng]["all"],:], dims=2), dims=2))
        end
    end
    Plots.histogram(s_vals, normalize=true, bins=0:1:15, label=nothing, color="gray")
    xlabel!("half decay (s)")
    ylabel!("fraction of encoding neurons")
end


function plot_neuron(fit_results, dataset, rng, neuron; plot_rng_only=true, plot_fit_idx=nothing, plot_rev=false, plot_stim=false, plot_size=(700,350), y_rng=(-1.5,3.5))
    max_t = plot_rng_only ? fit_results[dataset]["ranges"][rng][end] - fit_results[dataset]["ranges"][rng][1] + 1 : 1600
    rng_fit = plot_rng_only ? fit_results[dataset]["ranges"][rng] : 1:1600
    
    avg_timestep = fit_results[dataset]["avg_timestep"] / 60
    
    trace = fit_results[dataset]["trace_array"][neuron,rng_fit]
    all_rev = [t - rng_fit[1] + 1 for t in rng_fit if fit_results[dataset]["v"][t] < 0]
    Plots.plot()
    if plot_rev
        Plots.vline(avg_timestep .* all_rev, opacity=0.4, color=palette(:default)[2], label=nothing)
    end
    Plots.plot!(avg_timestep .* (rng_fit .- rng_fit[1]), trace, label=nothing, linewidth=2, color=palette(:default)[1], size=plot_size)
    
    tr1 = nothing
    tr2 = nothing
    if plot_fit_idx == :mle
        mle_est = deepcopy(fit_results[dataset]["trace_params"][rng, neuron, argmax(fit_results[dataset]["trace_scores"][rng, neuron, :]), :])
        mle_est[9] = -50
        tr1 = mle_est
        
        cmap = Gen.choicemap()
        update_cmap!(cmap, mle_est, nothing)

        (tr, _) = Gen.generate(unfold_nl8, (max_t, fit_results[dataset]["v"][rng_fit], fit_results[dataset]["θh"][rng_fit], fit_results[dataset]["P"][rng_fit]), cmap)
        fit = [tr[:chain => t => :y] for t=1:max_t]
        Plots.plot!(avg_timestep .* (rng_fit .- rng_fit[1]), fit, linewidth=2, label=nothing)
    elseif !isnothing(plot_fit_idx)
        for idx in plot_fit_idx
            params = deepcopy(fit_results[dataset]["sampled_trace_params"][rng, neuron, idx, :])
            params[9] = -100
            
            tr2 = params

            cmap = Gen.choicemap()
            update_cmap!(cmap, params, nothing)

            (tr, _) = Gen.generate(unfold_nl8, (max_t, fit_results[dataset]["v"][rng_fit], fit_results[dataset]["θh"][rng_fit], fit_results[dataset]["P"][rng_fit]), cmap)
            fit = [tr[:chain => t => :y] for t=1:max_t]
            Plots.plot!(avg_timestep .* (rng_fit .- rng_fit[1]), fit, linewidth=2, label=nothing)
        end
    end
    
    if plot_stim
        stim = fit_results[dataset]["ranges"][1][end]+1
        vline!([stim * avg_timestep], linewidth=3, label=nothing, color="red")
    end
    xlabel!("time (min)")
    ylabel!("neuron activity")
    yaxis!(y_rng)
end
