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
