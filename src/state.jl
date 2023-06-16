"""
    fit_state_classifier(x::Matrix{Float64}, t_stim::Int)

Fit a logistic regression model to classify whether a set of neurons contains "state" information about the indicator function of `t_stim`,
    based on neural data `x` and the time `t_stim` at which the hypothesized state change occurred. Returns the trained model and its accuracy.

# Arguments
- `x::Matrix{Float64}`: Input data matrix of size `(n_features, n_timepoints)`.
- `t_stim::Int`: Time of the hypothesized state change.

# Returns
- `model`: Trained logistic regression model.
- `accuracy`: Accuracy of the trained model.
"""
function fit_state_classifier(x::Matrix{Float64}, t_stim::Int)
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
