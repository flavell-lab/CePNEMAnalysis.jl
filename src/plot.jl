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
NOT YET IMPLEMENTED
"""
function plot_tsne()
    dataset = "2021-05-26-07"
    rng = 2
    style = :v
    label_ewma = true
    ewma_rng = 5
#     colors = (label_ewma ? [RGB.(0,0,1), RGB.(1,0,0), RGB.(0,1,0), RGB.(0.5,0,0.9)] : deepcopy(palette(:default)[1:4]))
    colors = [palette(:default)[1], palette(:default)[2], RGB.(0,1,0), RGB.(1,0,1)]# deepcopy(palette(:default)[1:4])
    append!(colors, colors * 0.5)

    sigmoid(x,t,a,b) = b * ((1 - a) + a * x / (x + t))
    push!(colors, RGB.(0.8,0.8,0.8))
    shapes = [:circle, :rtriangle, :ltriangle, :diamond]
    sizes = [4, 7, 7, 6]
    Plots.plot()
    used_idx = [false, false, false, false, false]
    used_shape_idx = [false, false, false, false]
    for i=1:length(n_encoding)
        n = n_encoding[i]
        label = nothing
        is_v = n in neuron_categorization[dataset][rng]["v"]["all"]
        
        
        is_θh = n in neuron_categorization[dataset][rng]["θh"]["all"]
        is_P = n in neuron_categorization[dataset][rng]["P"]["all"]
        if is_θh && is_P
            idx = 4
            label = "θh + P"
        elseif is_θh
            idx = 2
            label = "θh"
            if n in neuron_categorization[dataset][rng]["θh"]["rect_fwd_dorsal"] ||  # TODO: fix neurons with both dorsal and ventral tuning at different rectifications
                    n in neuron_categorization[dataset][rng]["θh"]["rect_rev_dorsal"]
                idx += 4
            end
        elseif is_P
            idx = 3
            label = "P"
            if n in neuron_categorization[dataset][rng]["P"]["rect_fwd_inh"] || 
                    n in neuron_categorization[dataset][rng]["P"]["rect_rev_inh"]
                idx += 4
            end
        elseif is_v
            label = "v only"
            idx = 1
        else
            idx = 5
        end
        is_fwd = n in neuron_categorization[dataset][rng]["v"]["fwd"]
        is_rev = n in neuron_categorization[dataset][rng]["v"]["rev"]
        if !is_v
            slabel = "no velocity"
            shape_idx = 1
        elseif is_fwd
            slabel = "forward"
            shape_idx = 2
        elseif is_rev
            slabel = "reversal"
            shape_idx = 3
        else
            slabel = "other v"
            shape_idx = 4
        end
        
        c = colors[idx] * (label_ewma ? sigmoid(median(compute_s.(fit_results[dataset]["sampled_trace_params"][rng,n,:,7])), 0, 0.5, 1.0) : 1)
#         if !used_idx[idx]
#             Plots.scatter!([tsne_dist[i,1]], [tsne_dist[i,2]], color=c, markerstrokecolor=c, markershape=shapes[shape_idx], label=label)
#             used_idx[idx] = true
#         elseif !used_shape_idx[shape_idx]
#             Plots.scatter!([tsne_dist[i,1]], [tsne_dist[i,2]], color=c, markerstrokecolor=c, markershape=shapes[shape_idx], label=slabel)
#             used_shape_idx[idx] = true
#         else
        Plots.scatter!([tsne_dist[i,1]], [tsne_dist[i,2]], label=nothing, color=c, markerstrokecolor=c, markersize=sizes[shape_idx], markershape=shapes[shape_idx], series_annotations=Plots.text.([n], :bottom, 7))
#         end
    end
    Plots.title!("t-SNE")#, xaxis=nothing, yaxis=nothing, framestyle=:box, size=(600,600))
#     savefig("/data1/prj_kfc/figure/2022-03-25-invertebrate/tsne.pdf")
#     savefig("/data1/prj_kfc/figure/2022-03-25-invertebrate/tsne.png")
    Plots.plot!()
end