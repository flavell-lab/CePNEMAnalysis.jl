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

include("data.jl")
include("dist.jl")
include("categorize.jl")
include("plot.jl")

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
    neuron_p_vals,
    categorize_neurons,
    # plot.jl
    make_deconv_heatmap
end # module
