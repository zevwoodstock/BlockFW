using FrankWolfe

"""
    BlockAdaptive(blocks, eta, tau, L_est, L_max)
    A line search method that adapts the step size for each block.

    # Arguments
    - `blocks::Vector{Int}`: The blocks to update.
    - `eta::T`: The eta parameter for the line search.
    - `tau::T`: The tau parameter for the line search.
    - `L_est::T`: The estimated value of the Lipschitz constant.
    - `L_max::T`: The maximum value of the Lipschitz constant.
"""


mutable struct BlockAdaptive{T} <: FrankWolfe.LineSearchMethod
    blocks::Vector{Int}
    eta::T
    tau::T
    L_est::T
    L_max::T
end

function FrankWolfe.perform_line_search(
    ls::BlockAdaptive,
    t,
    f,
    grad!,
    gradient,
    x,
    d,
    gamma_max,
    storage,
    memory_mode::FrankWolfe.MemoryEmphasis
)

    gamma = zeros(length(x.blocks))

    x_tilde = zero(x)
    ls.L_est = ls.eta * ls.L_est

    while ls.L_est < ls.L_max

        for i in ls.blocks
            gamma[i] = min(1, FrankWolfe.fast_dot(gradient.blocks[i], d.blocks[i]) / (ls.L_est * FrankWolfe.fast_dot(d.blocks[i], d.blocks[i])))
        end
        FrankWolfe.muladd_memory_mode(memory_mode, x_tilde, gamma, d)

        g_tilde = similar(gradient)
        grad!(g_tilde, x_tilde)

        if f(x) - f(x_tilde) - FrankWolfe.fast_dot(g_tilde, x - x_tilde) >= FrankWolfe.fast_dot(gradient - g_tilde, gradient - g_tilde) / (2 * ls.L_est)
            break
        else
            ls.L_est = ls.tau * ls.L_est
        end
    end

    if ls.L_est >= ls.L_max
        @warn("Line search failed to find a suitable step size")
    end

    return gamma
end

"""
    BlockLMO(blocks, lmos)
    A linear minimization oracle that calls only on the specified blocks.

    # Arguments
    - `blocks::Vector{Int}`: The blocks to update.
    - `lmos::LT`: The linear minimization oracles to use.
"""

mutable struct BlockLMO{LT<:Union{AbstractVector{FrankWolfe.LinearMinimizationOracle},Tuple{Vararg{FrankWolfe.LinearMinimizationOracle}}}} <: FrankWolfe.LinearMinimizationOracle
    blocks::Vector{Int}
    lmos::LT
end

function FrankWolfe.compute_extreme_point(lmo::BlockLMO, direction::FrankWolfe.BlockVector; v=nothing)
    v = zero(direction)
    for i in lmo.blocks
        v.blocks[i] = FrankWolfe.compute_extreme_point(lmo.lmos[i], direction.blocks[i])
    end
    return v
end


"""
    Callback-wrapper for block coordinate Frank-Wolfe.
"""
function make_block_selection_callback(callback::Union{Nothing,Function}, ls::BlockAdaptive, lmo::BlockLMO, update_order::FrankWolfe.BlockCoordinateUpdateOrder)

    cached_list_of_blocks = []

    return function new_callback(state, args...)

        # UpdateOrders can return multiple rounds of lists of blocks to update.
        # We cache the list of blocks to update and return the next round of blocks to update.
        if isempty(cached_list_of_blocks)
            cached_list_of_blocks = FrankWolfe.select_update_indices(update_order, state, zeros(length(state.x.blocks)))
        end
        blocks = pop!(cached_list_of_blocks)
        lmo.blocks = blocks
        ls.blocks = blocks

        if callback === nothing
            return true
        end

        return callback(state, args...)
    end
end


"""
    Adaption for iterate updates with per-block step sizes.
"""
function FrankWolfe.muladd_memory_mode(
    mem::FrankWolfe.MemoryEmphasis,
    x::FrankWolfe.BlockVector,
    gamma::Vector{T},
    d::FrankWolfe.BlockVector,
) where {T<:Real}
    @inbounds for i in eachindex(x.blocks)
        FrankWolfe.muladd_memory_mode(mem, x.blocks[i], gamma[i], d.blocks[i])
    end
    return x
end


"""
    adaptive_block_coordinate_frank_wolfe(f, grad!, lmos, x0; kwargs...)
    Adaptive block coordinate Frank-Wolfe.

    # Arguments
    - `f::Function`: The objective function to minimize.
    - `grad!: The gradient of the objective function.
    - `lmos::NTuple{N, FrankWolfe.LinearMinimizationOracle}`: The linear minimization oracles to use.
    - `x0::FrankWolfe.BlockVector`: The initial point.
    - `initial_blocks::Vector{Int}`: The initial blocks to update.
    - `update_order::FrankWolfe.BlockCoordinateUpdateOrder`: The update order to use.
    - `callback::Union{Nothing, Function}`: A callback function to call after each iteration.
    - `line_search::Union{Nothing, BlockAdaptive}`: The line search to use.
    - `eta::Real`: The eta parameter for the line search.
    - `tau::Real`: The tau parameter for the line search.
    - `L_max::Real`: The maximum value of the Lipschitz constant.
    - `L_est::Real`: The estimated value of the Lipschitz constant.
    - `kwargs...`: Additional keyword arguments to pass to the FrankWolfe.frank_wolfe function.
"""

function adaptive_block_coordinate_frank_wolfe(f, grad!, lmo::FrankWolfe.ProductLMO, x0; kwargs...)
    return adaptive_block_coordinate_frank_wolfe(f, grad!, lmo.lmos, x0; kwargs...)
end

function adaptive_block_coordinate_frank_wolfe(
    f,
    grad!,
    lmos::LT,
    x0::FrankWolfe.BlockVector;
    initial_blocks::Vector{Int}=collect(1:length(x0.blocks)),
    update_order::FrankWolfe.BlockCoordinateUpdateOrder=FrankWolfe.CyclicUpdate(),
    callback=nothing,
    line_search=nothing, # Listed here so we don't pass it in kwargs...
    eta::Real=0.9,
    tau::Real=2.0,
    L_max::Real=100.0,
    L_est::Real=1.0,
    kwargs...
) where {LT<:Union{AbstractVector{FrankWolfe.LinearMinimizationOracle},Tuple{Vararg{FrankWolfe.LinearMinimizationOracle}}}}


    ls = BlockAdaptive(initial_blocks, eta, tau, L_est, L_max)
    block_lmo = BlockLMO(initial_blocks, lmos)

    return FrankWolfe.frank_wolfe(
        f,
        grad!,
        block_lmo,
        x0;
        line_search=ls,
        callback=make_block_selection_callback(callback, ls, block_lmo, update_order),
        kwargs...
    )
end
