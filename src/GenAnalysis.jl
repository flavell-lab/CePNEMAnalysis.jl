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

export
    # data.jl
    load_gen_output

end # module
