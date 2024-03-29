"""
Makes a matrix of deconvolved neuron activity values from boundary points `deconvolved_activity` for either velocity and head angle (`axis=2`)
    or velocity and pumping (`axis=3`), compatible with making into a heatmap.
"""
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
Plots a deconvolved heatmap of the median particle.

# Arguments:
- `deconvolved_activity`: Deconvolved neural activity
- `v_ranges_plot`: Ranges to plot velocity
- `θh_ranges_plot`: Ranges to plot head angle
- `P_ranges_plot`: Ranges to plot pumping
- `axis`: Set to 1 for velocity only (plots velocity vs head angle in blue), 2 for head angle (velocity vs head angle in red), and 3 for pumping (velocity vs pumping in green)
- `res` (optional, default 200): Resolution of the plot
"""
function plot_deconvolved_heatmap(deconvolved_activity, v_ranges_plot, θh_ranges_plot, P_ranges_plot, axis; res=200)
    if axis == 1
        heatmap(v_rng, θh_rng, transpose(dropdims(median(make_deconv_heatmap(deconvolved_activity_plot[dataset][rng][neuron], 2, res=len), dims=1), dims=1)), c=cgrad([:black, :blue, :cyan]), clim=(-1,3))
    elseif axis == 2
        heatmap(v_rng, θh_rng, transpose(dropdims(median(make_deconv_heatmap(deconvolved_activity_plot[dataset][rng][neuron], axis, res=len), dims=1), dims=1)), c=cgrad([:black, :red, :magenta]), clim=(-1,3))
    elseif axis == 3
        heatmap(v_rng, P_rng, transpose(dropdims(median(make_deconv_heatmap(deconvolved_activity_plot[dataset][rng][neuron], axis, res=len), dims=1), dims=1)), c=cgrad([:black, :green, :turquoise]), clim=(-1,3))
    end
    xlabel!("velocity")
    if axis == 2
        ylabel!("head angle")
    else
        ylabel!("feeding")
    end
    Plots.plot!()
end

"""
    plot_deconvolved_neural_activity!(
        dataset, rngs, deconvolved_activity_plot, v_ranges_plot, θh_ranges_plot, P_ranges_plot, axis;
        res=200, plot_size=(700,350), init=true, len=200, pos=[100]
    )

Plots deconvolved neural activity.

# Arguments:
- `dataset`
- `rngs`: Ranges to plot
- `deconvolved_activity_plot`: Deconvolved neural activity, for plotting
- `v_ranges_plot`: velocity ranges
- `θh_ranges_plot`: head curvature ranges
- `P_ranges_plot`: feeding ranges
- `axis`: variable to plot. Can be 2 to plot velocity while varying head curvature, or 3 to plot feeding.
- `res` (optional, default 200): Resolution of plot
- `plot_size` (optional, default `(700,350)`): Size of plot.
- `init` (optional, default `true`): Initialize a new plot.
- `len`: Length to extrapolate other behaviors
- `pos`: Position of other behaviors to use
"""
function plot_deconvolved_neural_activity!(dataset, rngs, deconvolved_activity_plot, v_ranges_plot, θh_ranges_plot, P_ranges_plot, axis;
        res=200, plot_size=(700,350), init=true, len=200, pos=[100])

    v_rng = collect(range(v_ranges_plot[dataset][rngs[1]][1], v_ranges_plot[dataset][rngs[1]][2], length=Int(len/2)))
    append!(v_rng, range(v_ranges_plot[dataset][rngs[1]][3], v_ranges_plot[dataset][rngs[1]][4], length=Int(len/2)))


    P_rng = range(P_ranges_plot[dataset][1], P_ranges_plot[dataset][2], length=len)
    for rng in rngs
        if axis != 3
            hmap = make_deconvolved_heatmap(deconvolved_activity_plot[dataset][rng][neuron], axis, res=len)
        else
            hmap = permutedims(make_deconvolved_heatmap(deconv_P[neuron], axis, res=len), [1,3,2])
        end

        println(size(hmap))
        for idx in pos
            x = hmap[:,:,idx]
            med_x = dropdims(median(x, dims=1), dims=1)
            l = (idx == 1) ? "dorsal" : "ventral"
            if length(pos) == 1
                l = ""
            end
            if length(rngs) == 2 && dataset in heatstim_datasets
                l *= " " * ((rng == 1) ? "pre-stim" : "post-stim")
            end

            if axis != 3
                Plots.plot!(v_rng, med_x, ribbon=([med_x[i] - percentile(x[:,i], 5) for i=1:len], [percentile(x[:,i], 95) - med_x[i] for i=1:len]), label=l, legend=:topright)
            else
                Plots.plot!(P_rng, med_x, ribbon=([med_x[i] - percentile(x[:,i], 5) for i=1:len], [percentile(x[:,i], 95) - med_x[i] for i=1:len]), label=nothing, color=palette(:default)[3])
            end

        end
    end
    Plots.plot!()
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
- `modulation` (optional, default `false`): Whether you want to highlight special neurons by their response to neuromodulation/ other properties.
- `c_modulated` (optional, default `RGB.(0, 0, 0)`): The colors of special neurons listed in modulated_neurons.
- `modulated_datasets, modulated_ranges, modulated_neurons` (optional, default an empty set): The list of special neurons that you wish to highlight.
    `modulated_datasets` and `modulated_ranges` can be left empty, if so then the neuron will be matched across all datasets and ranges.
"""
function plot_tsne(tsne_dist, fit_results, dataset_ids_tsne, range_ids_tsne, neuron_ids_tsne, neuron_categorization, vars_plot; label_v_shapes=false, label_neurons=false, plot_axes=false,
        velocity_colors=[RGB.(0,0.9,1), RGB.(0.5,0.6,1), RGB.(0,0,1)], θh_colors=[RGB.(1,0.7,0.3), RGB.(1,0.3,0.7), RGB.(0.7,0,0)], P_colors=[RGB.(0,1,0), RGB.(0.7,1,0), RGB.(0,0.5,0)], 
        c_multiplex=RGB.(0.7,0.7,0.7), c_nonencoding=RGB.(0.8,0.8,0.8), shapes=[:circle, :rtriangle, :ltriangle, :diamond], sizes=[6,10,10,8], s_rng=[0,7], s_res=100, 
        modulation=false, c_modulated=RGB.(0,0,0), modulated_datasets=[], modulated_ranges=[], modulated_neurons=[])
            
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
                if n in neuron_categorization[dataset][rng]["θh"]["fwd_dorsal"] ||
                        n in neuron_categorization[dataset][rng]["θh"]["rev_dorsal"]
                    c = colors[2][3]
                    num_enc += 1
                end
                if n in neuron_categorization[dataset][rng]["θh"]["fwd_ventral"] ||
                    n in neuron_categorization[dataset][rng]["θh"]["rev_ventral"]
                    c = colors[2][1]
                    num_enc += 1
                end
                if num_enc != 1
                    c = colors[2][2]
                end
            elseif is_P && 3 in vars_plot
                num_enc = 0
                if n in neuron_categorization[dataset][rng]["P"]["fwd_inh"] || 
                        n in neuron_categorization[dataset][rng]["P"]["rev_inh"]
                    c = colors[3][3]
                    num_enc += 1
                end
                if n in neuron_categorization[dataset][rng]["P"]["fwd_act"] || 
                    n in neuron_categorization[dataset][rng]["P"]["rev_act"]
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
                
        if modulation
            for i=1:length(modulated_neurons)
                if n == modulated_neurons[i] && 
                        (length(modulated_datasets) == 0 || dataset == modulated_datasets[i]) && 
                        (length(modulated_ranges) == 0 || rng == modulated_ranges[i])
                    c = c_modulated
                    break
                end
            end
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
    plot_tau_histogram(
        fit_results, neuron_categorization; t_max=60, use_cdf=true, 
        percent=95, rngs_valid=nothing, behavior="all", behavior_subcat="all", label=nothing, legend=:topright
    )

Plots tau (half-decay) histogram of a single behavior.

# Arguments:
- `fit_results`: Gen fit results
- `neuron_categorization`: Categorization of each neuron
- `t_max` (optional, default 60): Maximum time point to plot
- `use_cdf` (optional, default true): Whether to use CDF format instead of PDF
- `percent` (optional, default 95): If using CDF format, credible range to show
- `rngs_valid` (optional): If set to a list of ranges, only attempt to use those ranges for computation.
- `behavior` (optional, default `all`): Behavior to use
- `behavior_subcat` (optional, default `all`): Subcategory of behavior to use
- `label`: Plot label
- `legend`: Plot legend location
"""
function plot_tau_histogram(fit_results, neuron_categorization; t_max=60, use_cdf=true, 
        percent=95, rngs_valid=nothing, behavior="all", behavior_subcat="all", label=nothing, legend=:topright)
    s_vals = []
    s_vals_min = []
    s_vals_max = []
    delta = (100-percent)/2
    for dataset in keys(fit_results)
        if isnothing(rngs_valid)
            rngs_valid_use = 1:length(fit_results[dataset]["ranges"])
        else
            rngs_valid_use = rngs_valid
        end
        for n in 1:fit_results[dataset]["num_neurons"]
            if behavior == "all"
                rngs_usable = [rng for rng in rngs_valid_use if n in neuron_categorization[dataset][rng][behavior]]
            else
                rngs_usable = [rng for rng in rngs_valid_use if n in neuron_categorization[dataset][rng][behavior][behavior_subcat]]
            end
                
            if length(rngs_usable) == 0
                continue
            end
            sv_min = []
            sv = []
            sv_max = []
            for rng in rngs_usable
                append!(sv_min, percentile(fit_results[dataset]["sampled_tau_vals"][rng,n,:], delta))
                append!(sv, median(fit_results[dataset]["sampled_tau_vals"][rng,n,:]))
                append!(sv_max, percentile(fit_results[dataset]["sampled_tau_vals"][rng,n,:], 100-delta))
            end
            push!(s_vals_min, median(sv_min))
            push!(s_vals, median(sv))
            push!(s_vals_max, median(sv_max))
        end
    end
    if use_cdf
        r = 0:0.1:t_max
        mn = [sum(s_vals_min .< i) ./ length(s_vals) for i=r]
        m = [sum(s_vals .< i) ./ length(s_vals) for i=r]
        mx = [sum(s_vals_max .< i) ./ length(s_vals) for i=r]
        
        Plots.plot!(r, m, ribbon=(m.-mn, mx.-m), label=label, ylim=(0,1), legend=legend)
        ylabel!("cumulative fraction of encoding neurons")
    else
        Plots.histogram(s_vals, normalize=true, bins=0:1:t_max, label=nothing, color="gray")
    end
    xlabel!("half decay (s)")
    ylabel!("fraction of encoding neurons")
end


"""
Plots a neuron and model fits to that neuron.

# Arguments:
- `fit_results`: Dictionary of all model fit results.
- `dataset`: Dataset corresponding to the neuron
- `rng`: Range where the neuron was fit
- `neuron`: Neuron
- `plot_rng_only` (optional, default `true`): Plot only the range of the fit (vs the entire time range)
- `plot_fit_idx` (optional, default `nothing`): Plot a Gen fit to the neuron.
    - If set to an array of numbers, plot the particles with indices in that array.
    - If set to `:mle`, plot the maximum-likelihood particle. Only available for SMC fits, not MCMC fits.
    - If set to `:median`, plot the median trace.
- `use_heatmap` (optional, default `false`): If `plot_fit_idx` is an array, plot the fits as a heatmap instead of as lines.
- `heatmap_hist_step` (optional, default `0.01`): Histogram `y`-step
- `plot_rev` (optional, default `false`): Plot reversal events.
- `plot_stim` (optional, default `false`): Plot the heat stim.
- `plot_size` (optional, default `(700,350)`): Size of the plot
- `x_rng` (optional, default `0:3:18`): `x`-range of the plot
- `y_rng` (optional, default `-2.:1.:4.`): `y`-range of the plot
- `linewidth` (optional, default 2): Width of neuron line
- `contrast` (optional, default 99): If `use_heatmap` is true, contrast of heatmap
- `idx_split` (optional, default `[1:1600]`): Ranges to fit separately (for resetting EWMA purposes). Must be contiguous.
"""
function plot_neuron(fit_results::Dict, dataset::String, rng::Int, neuron::Int; plot_rng_only::Bool=true, plot_fit_idx=nothing, use_heatmap::Bool=false, 
        heatmap_hist_step::Real=0.01, plot_rev::Bool=false, plot_stim::Bool=false, plot_size=(700,350), y_rng=-2.:1.:4., x_rng=0:3:18, linewidth=2, contrast=99, idx_split=[1:1600])
    max_t = plot_rng_only ? fit_results[dataset]["ranges"][rng][end] - fit_results[dataset]["ranges"][rng][1] + 1 : idx_split[end][end] - idx_split[1][1] + 1


    rngs_fit = plot_rng_only ? [fit_results[dataset]["ranges"][rng]] : idx_split

    if length(rngs_fit) > 1
        for i=2:length(rngs_fit)
            @assert(rngs_fit[i-1][end] + 1 == rngs_fit[i][1], "Fit ranges must be contiguous.")
        end
    end

    avg_timestep = fit_results[dataset]["avg_timestep"] / 60

    trace = fit_results[dataset]["trace_array"][neuron,:]
    all_rev = [t for t in rngs_fit[1][1]:rngs_fit[end][end] if fit_results[dataset]["v"][t] < 0]
    Plots.plot()

    tr1 = nothing
    tr2 = nothing

    x_rng_plot = avg_timestep .* (rngs_fit[1][1]:rngs_fit[end][end])


    if plot_fit_idx in [:mle, :median]
        if plot_rev
            Plots.vline!(avg_timestep .* all_rev, opacity=0.1+0.3*plot_rng_only, color=palette(:tab10)[2], label=nothing)
        end
        Plots.plot!(x_rng_plot, trace[rngs_fit[1][1]:rngs_fit[end][end]], label=nothing, linewidth=linewidth, color=palette(:tab10)[1], size=plot_size)
    end

    color_idx = 2
    if plot_rev && !(plot_fit_idx in [:mle, :median])
        color_idx = 3
    end
    if plot_fit_idx == :mle
        mle_est = fit_results[dataset]["trace_params"][rng, neuron, argmax(fit_results[dataset]["trace_scores"][rng, neuron, :]), 1:8]
        f = zeros(max_t)
        f0 = rngs_fit[1][1] - 1
        for rng_fit = rngs_fit
            params = deepcopy(mle_est)
            if !plot_rng_only
                params[6] = trace[rng_fit[1]]
            end
            f[rng_fit .- f0] .= model_nl8(rng_fit[end] - rng_fit[1] + 1, params..., fit_results[dataset]["v"][rng_fit], fit_results[dataset]["θh"][rng_fit], fit_results[dataset]["P"][rng_fit])
        end
        Plots.plot!(x_rng_plot, f, linewidth=2, label=nothing, color=palette(:tab10)[color_idx])
    elseif !isnothing(plot_fit_idx)
        plot_idx = plot_fit_idx == :median ? (1:size(fit_results[dataset]["sampled_trace_params"], 3)) : plot_fit_idx
        if use_heatmap || plot_fit_idx == :median
            y_min = y_rng[1]
            y_max = y_rng[end]
            y_fit_array = zeros(max_t, length(plot_idx))

            for (idx, i) = enumerate(plot_idx)
                f = zeros(max_t)
                f0 = rngs_fit[1][1] - 1
                for rng_fit = rngs_fit
                    params = deepcopy(fit_results[dataset]["sampled_trace_params"][rng,neuron,i,1:8])
                    if !plot_rng_only
                        params[6] = trace[rng_fit[1]]
                    end
                    f[rng_fit .- f0] .= model_nl8(rng_fit[end] - rng_fit[1] + 1, params..., fit_results[dataset]["v"][rng_fit], fit_results[dataset]["θh"][rng_fit], fit_results[dataset]["P"][rng_fit])
                end
                y_fit_array[:, idx] .= f
            end

            if plot_fit_idx == :median
                m = dropdims(median(y_fit_array, dims=2), dims=2)
                Plots.plot!(x_rng_plot, m, linewidth=2, label=nothing, color=palette(:tab10)[color_idx])
            else
                y_bins = y_min:heatmap_hist_step:y_max
                y_hist_array = zeros(max_t, length(y_bins) - 1)
                for t = 1:max_t
                    hist_fit = fit(Histogram, y_fit_array[t,:],  y_bins)
                    y_hist_array[t,:] = hist_fit.weights
                end
                heatmap!(x_rng_plot, y_bins[1:end-1] .+ 0.005, y_hist_array', 
                        c=cgrad([:white, palette(:tab10)[color_idx], :black]), 
                        clim=(percentile(reshape(y_hist_array, length(y_hist_array)),(100-contrast)), percentile(reshape(y_hist_array, length(y_hist_array)),contrast)), colorbar=nothing, left_margin=5mm)
            end
        else
            for idx in plot_fit_idx
                params = deepcopy(fit_results[dataset]["sampled_trace_params"][rng, neuron, idx, 1:8])
                f = zeros(max_t)
                f0 = rngs_fit[1][1] - 1
                for rng_fit = rngs_fit
                    if !plot_rng_only
                        params[6] = trace[1]
                    end
                    f[rng_fit .- f0] .= model_nl8(rng_fit[end] - rng_fit[1] + 1, params..., fit_results[dataset]["v"][rng_fit], fit_results[dataset]["θh"][rng_fit], fit_results[dataset]["P"][rng_fit])
                end
                Plots.plot!(x_rng_plot, f, linewidth=2, label=nothing, color=palette(:tab10)[color_idx])
                color_idx += 1
            end
        end
    end

    if !(plot_fit_idx in [:mle, :median])
        if plot_rev
            Plots.vline!(avg_timestep .* all_rev, opacity=0.1+0.3*plot_rng_only, color=palette(:tab10)[2], label=nothing)
        end
        Plots.plot!(x_rng_plot, trace[rngs_fit[1][1]:rngs_fit[end][end]], label=nothing, linewidth=linewidth, color=palette(:tab10)[1], size=plot_size)
    end

    if plot_stim
        stim = fit_results[dataset]["ranges"][1][end]+1
        vline!([stim * avg_timestep], linewidth=3, label=nothing, color="red")
    end
    Plots.plot!(xaxis = ("time (min)", (x_rng[1], x_rng[end]), x_rng, font(7, "Arial")),
        yaxis = ("neuron activity (AU)", (y_rng[1], y_rng[end]), y_rng, font(7, "Arial"))), x_rng_plot
end

"""
    plot_posterior_heatmap!(fit_results, dataset, rng, neuron, param1, param2; init=true, color=palette(:default)[2], x_rng=-3:0.1:3, y_rng=-3:0.1:3, rgb=false)

Plots the heatmap of the projection of posterior particles of a neuron into a 2D susbspace.

# Arguments:
- `fit_results`: Dictionary of Gen fit results.
- `dataset`: Dataset containing the neuron.
- `rng`: Range where the neuron was fit.
- `neuron`: Neuron
- `param1`: First parameter to plot (as an index 1 through 9)
- `param2`: Second parameter to plot (as an index 1 through 9)
- `c1rng`: First parameter range
- `c2rng`: Second parameter range
- `init` (optional, default `true`): Initialize a new plot, rather than overlaying on top of a preexisting plot
- `color` (optional, default `palette(:default)[2]`): Color of the heatmap
- `x_rng` (optional, default `-3:0.1:3`): `x`-axis range
- `y_rng` (optional, default `-3:0.1:3`): `y`-axis range
- `rgb` (optional, default `false`): If set, return RGB histogram and x and y ranges, instead of a plot.
"""
function plot_posterior_heatmap!(fit_results, dataset, rng, neuron, param1, param2; init=true, color=palette(:default)[2], x_rng=-3:0.1:3, y_rng=-3:0.1:3, rgb=false)
    if init
        Plots.plot()
    end
    c11 = fit_results[dataset]["sampled_trace_params"][rng,neuron,:,param1]
    c12 = fit_results[dataset]["sampled_trace_params"][rng,neuron,:,param2]    
 

    hist_fit = fit(Histogram, (c11,c12), (x_rng, y_rng))
    if !rgb
        return Plots.heatmap!(hist_fit.weights, c=cgrad([:white, color, :black]), xaxis=nothing, yaxis=nothing, framestyle=:box, colorbar=nothing, size=(500,500))
    else
        return hist_fit.weights ./ maximum(hist_fit.weights) .* color, x_rng, y_rng
    end
end

"""
Plots posterior RGB.

# Arguments:
- `posterior`: RGB posterior.
- `x_rng`: x-range of the posterior
- `y_rng`: y-range of the posterior
- `param_x`: x parameter
- `param_y`: y parameter
"""
function plot_posterior_rgb(posterior, x_rng, y_rng, param_x, param_y)
    params = [param_x, param_y]
    labels = ["", ""]
    for (i,param)=enumerate(params)
        if param == 1
            labels[i] = "velocity threshold"
        elseif param == 2
            labels[i] = "velocity"
        elseif param == 3
            labels[i] = "head curvature"
        elseif param == 4
            labels[i] = "feeding"
        elseif param == 5
            labels[i] = "velocity rectification"
        elseif param == 6
            labels[i] = "initial neural activity"
        elseif param == 7
            labels[i] = "timescale"
        elseif param == 8
            labels[i] = "baseline"
        elseif param == 9
            labels[i] = "model uncertainty timescale"
        elseif param == 10
            labels[i] = "uncertainty level"
        elseif param == 11
            labels[i] = "noise level"
        end
    end
    Plots.plot(x_rng, y_rng, posterior, reverse=true)
    xlabel!(labels[1])
    ylabel!(labels[2])
end

"""
    plot_arrow!(arrow_start::Tuple{Real, Real}, arrow_end::Tuple{Real, Real}, arrow_color::Color, arrow_width::Real, arrow_length::Real)

Plots an arrow from `arrow_start` to `arrow_end` with a given `arrow_color`, `arrow_width`, and `arrow_length`.

# Arguments:
- `arrow_start::Tuple{Real, Real}`: Starting point of the arrow.
- `arrow_end::Tuple{Real, Real}`: Ending point of the arrow.
- `arrow_color::Color`: Color of the arrow.
- `arrow_width::Real`: Width of the arrow.
- `arrow_length::Real`: Length of the arrow.
"""
function plot_arrow!(arrow_start::Tuple{Real, Real}, arrow_end::Tuple{Real, Real}, arrow_color::Color, arrow_width::Real, arrow_length::Real)
    Plots.plot!([arrow_start[1], arrow_end[1]], [arrow_start[2], arrow_end[2]], color=arrow_color, linewidth=arrow_width, label=nothing)
    d = sqrt((arrow_end[1] - arrow_start[1])^2 + (arrow_end[2] - arrow_start[2])^2)
    θ = asin((arrow_start[2] - arrow_end[2]) / d)
    if arrow_end[1] - arrow_start[1] > 0
        θ = π-θ
    end
    arrow_pt_1 = arrow_end .+ arrow_length .* (cos(θ + π/4), sin(θ + π/4))
    Plots.plot!([arrow_end[1], arrow_pt_1[1]], [arrow_end[2], arrow_pt_1[2]], color=arrow_color, linewidth=arrow_width, label=nothing)
    arrow_pt_2 = arrow_end .+ arrow_length .* (cos(θ - π/4), sin(θ - π/4))
    Plots.plot!([arrow_end[1], arrow_pt_2[1]], [arrow_end[2], arrow_pt_2[2]], color=arrow_color, linewidth=arrow_width, label=nothing)
end    

"""
    color_to_rgba(color::Color, alpha::Real) -> Tuple{Float64, Float64, Float64, Float64}

Converts a `Color` object to an RGBA tuple with the given alpha value.

# Arguments:
- `color`: The `Color` object to convert.
- `alpha`: The alpha value to use for the RGBA tuple.

# Returns:
- A tuple of four `Float64` values representing the RGBA values of the input `Color` object with the given alpha value.
"""
function color_to_rgba(color::Color, alpha::Real)::Tuple{Float64, Float64, Float64, Float64}
    return (color.r, color.g, color.b, alpha)
end

"""
    plot_colorbar(rng_min::Real, rng_max::Real, other_ticks::Vector, cmap, n_colors::Integer, figsize::Tuple{Real, Real})

Plots a colorbar with a gradient of colors ranging from `rng_min` to `rng_max` with `n_colors` colors. The `other_ticks` argument is a vector of additional ticks to be displayed on the colorbar. The `cmap` argument is a `ColorMap` object that specifies the color scheme to be used. The `n_colors` argument is an `Integer` specifying the number of colors to be used in the gradient. The `figsize` argument is a tuple of two `Real` values that specifies the size of the figure.

# Arguments:
- `rng_min::Real`: The minimum value of the range of values to be displayed on the colorbar.
- `rng_max::Real`: The maximum value of the range of values to be displayed on the colorbar.
- `other_ticks::Vector`: A vector of additional ticks to be displayed on the colorbar.
- `cmap`: A ColorMap that specifies the color scheme to be used.
- `n_colors::Integer`: The number of colors to be used in the gradient.
- `figsize::Tuple{Real, Real}`: A tuple of two `Real` values that specifies the size of the figure.

# Returns:
- Nothing. The function is called for its side effects of plotting the colorbar.
"""
function plot_colorbar(rng_min::Real, rng_max::Real, other_ticks::Vector, cmap, n_colors::Integer, figsize::Tuple{Real, Real})
    # Create an array with the range of input values for the colormap
    gradient = reshape(range(rng_min, stop=rng_max, length=n_colors), 1, n_colors)

    # Plot the gradient and the colorbar
    fig, ax = subplots(figsize=figsize)
    img = ax.imshow(gradient, cmap=cmap, aspect="auto", origin="lower")
    colorbar(img, ax=ax, cmap=cmap, ticks=[rng_min, other_ticks..., rng_max])

    gca().set_visible(false)
end

"""
    get_color_from_palette(value::Real, min_val::Real, max_val::Real, cmap::ColorMap) -> Color

Returns the color corresponding to a given value in a colormap. The colormap is defined by the `cmap` argument, which is a `ColorMap` object. The `value` argument is the value for which the corresponding color is to be found. The `min_val` and `max_val` arguments define the range of values that the colormap spans.

# Arguments:
- `value`: The value for which the corresponding color is to be found.
- `min_val`: The minimum value of the range of values that the colormap spans.
- `max_val`: The maximum value of the range of values that the colormap spans.
- `cmap`: A `ColorMap` object that defines the colormap.

# Returns:
- A `Color` object corresponding to the given value in the colormap.
"""
function get_color_from_palette(value::Real, min_val::Real, max_val::Real, cmap::ColorMap)
    if value < min_val
        value = min_val
    elseif value > max_val
        value = max_val
    end

    # Normalize the value to the range [0, 1]
    normalized_value = (value - min_val) / (max_val - min_val)

    # Get the RGB color from the colormap
    color = cmap(normalized_value)

    return color
end
