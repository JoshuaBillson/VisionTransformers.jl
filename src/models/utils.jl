_per_layer_drop_path(drop_rate, depth::Int) = LinRange(0, drop_rate, depth) |> collect
function _per_layer_drop_path(drop_rate, depths::Vector{Int})
    path_drop_rates = []
    linear_drop_rates = collect(LinRange(0, drop_rate, sum(depths)))
    lower_bound = 1
    for upper_bound in cumsum(depths)
        push!(path_drop_rates, linear_drop_rates[lower_bound:upper_bound])
        lower_bound = upper_bound + 1
    end
    return path_drop_rates
end