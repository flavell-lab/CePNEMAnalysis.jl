module GenAnalysis

using FlavellBase
using EncoderModelGen
using HDF5
using Gen
using Statistics
using StatsBase
using ProgressMeter
using Plots
using TSne
using ColorSchemes
using MultipleTesting

include("data.jl")
include("dist.jl")
include("categorize.jl")
include("plot.jl")
include("tsne.jl")

export
    # data.jl
    load_gen_output,
    # dist.jl
    update_cmap!,
    evaluate_pdf_xgiveny!,
    compute_dists,
    kl_div,
    overlap_index,
    prob_P_greater_Q,
    # categorize.jl
    deconvolved_model_nl7b,
    compute_range,
    get_deconvolved_activity,
    make_deconvolved_lattice,
    neuron_p_vals,
    categorize_neurons,
    categorize_all_neurons,
    detect_encoding_changes,
    # tsne.jl
    make_distance_matrix,
    compute_tsne,
    # plot.jl
    make_deconvolved_heatmap,
    plot_tsne,
    plot_tau_histogram,
    plot_neuron
end # module
