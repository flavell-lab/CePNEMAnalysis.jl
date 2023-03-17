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
