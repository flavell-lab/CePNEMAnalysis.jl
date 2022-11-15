module GenAnalysis

using FlavellBase
using EncoderModelGen
using HDF5
using Gen
using Statistics
using StatsBase
using ProgressMeter
using Plots
using Plots.PlotMeasures
using TSne
using ColorSchemes
using MultipleTesting
using MultivariateStats
using AnalysisBase
using FlavellConstants, ANTSUNData

include("data.jl")
include("dist.jl")
include("encoding-categorization.jl")
include("encoding-change.jl")
include("decode.jl")
include("plot.jl")
include("tsne.jl")
include("cluster.jl")
include("pca.jl")
include("strength.jl")
include("tuning.jl")

export
    # data.jl
    load_gen_output,
    get_pumping_ranges,
    # dist.jl
    update_cmap!,
    evaluate_pdf_xgiveny!,
    compute_dists,
    kl_div,
    overlap_index,
    prob_P_greater_Q,
    project_posterior,
    # encoding-categorization.jl
    VALID_V_COMPARISONS,
    deconvolved_model_nl8,
    compute_range,
    get_deconvolved_activity,
    make_deconvolved_lattice,
    neuron_p_vals,
    categorize_neurons,
    categorize_all_neurons,
    subcategorize_all_neurons!,
    get_neuron_category,
    get_enc_stats,
    get_enc_stats_pool,
    get_consistent_neurons,
    encoding_summary_stats,
    # encoding-change.jl
    detect_encoding_changes,
    correct_encoding_changes,
    get_enc_change_stats,
    encoding_summary_stats,
    get_subcats,
    get_enc_change_cat_p_vals,
    get_enc_change_cat_p_vals_dataset,
    get_enc_change_category,
    # tsne.jl
    make_distance_matrix,
    compute_tsne,
    # plot.jl
    CET_L13,
    make_deconvolved_heatmap,
    plot_deconvolved_heatmap,
    plot_deconvolved_neural_activity!,
    plot_tsne,
    plot_tau_histogram,
    plot_neuron,
    make_umap_rgb,
    plot_posterior_heatmap!,
    plot_posterior_rgb,
    plot_arrow!,
    color_to_rgba,
    # pca.jl
    extrapolate_neurons,
    make_distance_matrix,
    invert_id,
    invert_array,
    # cluster.jl
    cluster_dist,
    dendrocolor,
    # strength.jl
    get_relative_encoding_strength_mt,
    # tuning.jl
    get_forwardness,
    get_dorsalness,
    get_tuning_strength,
    # decode.jl
    fit_decoder,
    compute_variance_explained
end # module
