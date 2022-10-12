"""
Computes the relative encoding strength of the three behaviors, together with standard deviations of full and deconvolved model fits.

# Arguments
- `fit_results`: Gen fit results.
- `dataset`: Dataset to use
- `rng`: Range in that dataset to use
- `neuron`: Neuron to use
- `max_idx` (optional, default `10001`): Maximum Gen posterior sample index.
- `dataset_mapping` (optional, default `nothing`): Mapping to use different behavioral dataset.
"""
function get_relative_encoding_strength_mt(fit_results, dataset, rng, neuron; max_idx=10001, dataset_mapping=nothing)
    dataset_fit = fit_results[dataset]
    ps_fit = dataset_fit["sampled_trace_params"]
    rng_t = dataset_fit["ranges"][rng]
    max_t = length(rng_t)
    
    dset = (isnothing(dataset_mapping) ? dataset : dataset_mapping[dataset])
    b_v = fit_results[dset]["v"][rng_t]
    b_θh = fit_results[dset]["θh"][rng_t]
    b_P = fit_results[dset]["P"][rng_t]
    b_null = zeros(max_t)
    
    std_deconv = zeros(max_idx)
    std_deconv_v = zeros(max_idx)
    std_deconv_θh = zeros(max_idx)
    std_deconv_P = zeros(max_idx)

    std_full = zeros(max_idx)
    std_full_nops6 = zeros(max_idx)
    std_v = zeros(max_idx)
    std_θh = zeros(max_idx)
    std_P = zeros(max_idx)

    err_v = zeros(max_idx)
    err_θh = zeros(max_idx)
    err_P = zeros(max_idx)
    
    model_deconv = zeros(max_idx, max_t)
    model_deconv_v = zeros(max_idx, max_t)
    model_deconv_θh = zeros(max_idx, max_t)
    model_deconv_P = zeros(max_idx, max_t)
        
     @sync Threads.@threads for idx=1:max_idx
        ps = deepcopy(ps_fit[rng,neuron,idx,1:8])


        model_full = model_nl8(max_t, ps..., b_v, b_θh, b_P)
        model_nov = model_nl8(max_t, ps..., b_null, b_θh, b_P)
        model_noθh = model_nl8(max_t, ps..., b_v, b_null, b_P)
        model_noP = model_nl8(max_t, ps..., b_v, b_θh, b_null)

        ps[6] = 0 # delete initial condition so as not to contaminate deconvolution computation

        model_full_nops6 = model_nl8(max_t, ps..., b_v, b_θh, b_P)
        model_v = model_nl8(max_t, ps..., b_v, b_null, b_null)
        model_θh = model_nl8(max_t, ps..., b_null, b_θh, b_null)
        model_P = model_nl8(max_t, ps..., b_null, b_null, b_P)

        
        for i_t = 1:length(rng_t)
            model_deconv[idx, i_t] = deconvolved_model_nl8(ps, b_v[i_t], b_θh[i_t], b_P[i_t])
            model_deconv_v[idx, i_t] = deconvolved_model_nl8(ps, b_v[i_t], b_null[i_t], b_null[i_t])
            model_deconv_θh[idx, i_t] = deconvolved_model_nl8(ps, b_null[i_t], b_θh[i_t], b_null[i_t])
            model_deconv_P[idx, i_t] = deconvolved_model_nl8(ps, b_null[i_t], b_null[i_t], b_P[i_t])
        end        
        
        std_full[idx] = std(model_full)
        std_v[idx] = std(model_v)
        std_θh[idx] = std(model_θh)
        std_P[idx] = std(model_P)

        std_full_nops6[idx] = std(model_full_nops6)
        std_deconv[idx] = std(model_deconv[idx,:])
        std_deconv_v[idx] = std(model_deconv_v[idx,:])
        std_deconv_θh[idx] = std(model_deconv_θh[idx,:])
        std_deconv_P[idx] = std(model_deconv_P[idx,:])

        ev = cost_mse(model_full, model_nov)
        eθh = cost_mse(model_full, model_noθh)
        eP = cost_mse(model_full, model_noP)

        s = ev + eθh + eP
        err_v[idx] = ev / s
        err_θh[idx] = eθh / s
        err_P[idx] = eP / s
    end
    
    return err_v, err_θh, err_P, std_full, std_full_nops6, std_v, std_θh, std_P, std_deconv, std_deconv_v, std_deconv_θh, std_deconv_P
end
