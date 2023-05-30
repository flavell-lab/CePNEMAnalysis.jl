"""
Evaluates a Gaussian kernel at position `x`` with standard deviation `sigma`.
"""
function gaussian_kernel(x::Real; sigma::Float64=1.0)
    return exp(-x^2 / (2 * sigma^2)) / sqrt(2 * pi * sigma^2)
end

"""
Convolves a vector `x` with a kernel vector `k`.
"""
function convolve(x::Vector{Float64}, k::Vector{Float64})
    n = length(x)
    m = length(k)
    y = zeros(n + m - 1)
    for i in 1:n
        for j in 1:m
            y[i + j - 1] += x[i] * k[j]
        end
    end
    return y
end

"""
    Fits a GLM that determines how well the data vector `x` predicts the indicator function of the state indicated by `t_stim` (ie: time greater than `t_stim`).
    Returns the model and the accuracy of the model.
"""
function fit_state_classifier(x, t_stim)
    y = zeros(Int64, size(x,2))
    y[1:t_stim] .= 0
    y[t_stim+1:end] .= 1

    w = zeros(Float64, size(x,2))
    w[1:t_stim] .= length(w) - t_stim
    w[t_stim+1:end] .= t_stim

    w = w ./ mean(w)

    # Convert the input data and target vector to a DataFrame

    # if size(x, 1) > 1
    #     println(DataFrame(convert(Matrix{Float64}, transpose(x)), :auto))
    # end
    df = hcat(DataFrame(y=y, w=w), DataFrame(convert(Matrix{Float64}, transpose(x)), :auto))

    # Create the logistic regression model
    model = glm(term(:y) ~ sum(term(t) for t in names(df) if (t != "y") && (t != "w")), df, Binomial(), LogitLink(), wts=df.w)

    # Use the model to make predictions for new games
    predictions = GLM.predict(model, df)

    # Converting probability score to classes
    prediction_class = [if x < 0.5 0 else 1 end for x in predictions];

    accuracy_0 = mean([prediction_class[i] == df.y[i] for i in 1:length(df.y) if df.y[i] == 0])
    accuracy_1 = mean([prediction_class[i] == df.y[i] for i in 1:length(df.y) if df.y[i] == 1])
    accuracy = mean([accuracy_0, accuracy_1])

    return (model, accuracy)
end

"""
    Finds all neurons with the given encodings to behavior. Performs Benjamini-Hochberg multiple-hypothesis correction across multiple time ranges in the dataset, and performs Bonferroni correction
        across the behaviors.

    # Arguments:
    - `fit_results::Dict`: Dictionary of CePNEM fit results.
    - `analysis_dict::Dict`: Dictionary of CePNEM analysis results.
    - `beh::String`: Behavior to analyze.
    - `sub_behs::Union{Nothing,Vector{String}}`: Sub-behaviors to analyze. Can set to `nothing` for behaviors with no sub-behaviors (for example, `all`)
    - `p::Float64` (optional, default `0.05`): p-value threshold for significance.
"""
function get_all_neurons_with_feature(fit_results::Dict, analysis_dict::Dict, beh::String, sub_behs::Union{Nothing,Vector{String}}; p::Float64=0.05)
    traces_use = Dict()
    neurons_use = Dict()
    for dataset in keys(fit_results)
        neurons_use[dataset] = Int32[]

        for neuron in 1:size(fit_results[dataset]["trace_array"],1)
            use = false
            if isnothing(sub_behs)
                pval = minimum(adjust([analysis_dict["neuron_p"][dataset][rng][beh][neuron] for rng = 1:length(fit_results[dataset]["ranges"])], BenjaminiHochberg()))
                if pval < p
                    use = true
                end
            else
                for sub_beh in sub_behs
                    pval = length(sub_behs)*minimum(adjust([analysis_dict["neuron_p"][dataset][rng][beh][sub_beh][neuron] for rng = 1:length(fit_results[dataset]["ranges"])], BenjaminiHochberg()))
                    if pval < p
                        use = true
                        break
                    end
                end
            end
            if use
                push!(neurons_use[dataset], neuron)
            end
        end

        traces_use[dataset] = fit_results[dataset]["trace_array"][neurons_use[dataset], :]
    end
    return neurons_use, traces_use
end

"""
    correct_name(neuron_name)

Corrects the name of a neuron by removing the "0" substring from the name if it contains "DB0" or "VB0".
This causes the neuron name to be compatible with the connectome files.

# Arguments:
- `neuron_name::String`: The name of the neuron to be corrected.

# Returns:
- `neuron_name::String`: The corrected name of the neuron.
"""
function correct_name(neuron_name::String)
    if occursin("DB0", neuron_name)
        neuron_name = neuron_name[1:2]*neuron_name[4:end]
    end
    if occursin("VB0", neuron_name)
        neuron_name = neuron_name[1:2]*neuron_name[4:end]
    end
    return neuron_name
end

"""
find_peaks(neural_activity::Vector{Float64}, threshold::Float64)

Find the peaks in a vector of neural activity above a given threshold.
This method is most useful for finding spikes in the activity of spiking neurons.

# Arguments:
- `neural_activity::Vector{Float64}`: A vector of neural activity.
- `threshold::Float64`: The threshold above which to consider a value a peak.

# Returns:
- `peaks::Vector{Int}`: A vector of indices of the peaks in the neural activity.
- `peak_heights::Vector{Float64}`: A vector of the heights of the peaks in the neural activity.
"""
function find_peaks(neural_activity::Vector{Float64}, threshold::Float64)
    peaks = Int[]
    peak_heights = Float64[]
    over_threshold = false
    curr_peak = (-1, -Inf)
    for i in 1:length(neural_activity)
        if neural_activity[i] > threshold && !over_threshold
            over_threshold = true
            curr_peak = (i, neural_activity[i])
        elseif neural_activity[i] < threshold && over_threshold
            over_threshold = false
            push!(peaks, curr_peak[1])
            push!(peak_heights, curr_peak[2])
            curr_peak = (-1, -Inf)
        elseif neural_activity[i] > curr_peak[2] && over_threshold
            curr_peak = (i, neural_activity[i])
        end
    end
    return peaks, peak_heights
end