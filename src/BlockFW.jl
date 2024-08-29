using FrankWolfe
using LinearAlgebra
using Plots


struct LazyUpdate <: FrankWolfe.BlockCoordinateUpdateOrder 
	#Update order discussed in "Flexible block-iterative
	#analysis for the Frank-Wolfe algorithm," by Braun,
	#Pokutta, & Woodstock (2024). 
	#'lazy_block' is an index of a computationally expensive
	#block;
	#'refresh_rate' describes the frequency at which we perform
	#a full activation; and
	#'block_size' describes the number of "faster" blocks
	#(i.e., those excluding 'lazy_block') activated during each
	#of the "faster" iterations. If  'block_size' is
	#unspecified, this defaults to 1.
	#During the "faster" iterations, a uniformly random subset
	#of the faster blocks (i.e., those excluding 'lazy_block')
	#is updated.
	#*Important* This methodology is currently only
	#proven to work with FrankWolfe.Shortstep linesearches and
	#a (not-yet implemented) adaptive method; see the article
	#for details.

    lazy_block::Int
    refresh_rate::Int
    block_size::Int
end

function LazyUpdate(lazy_block::Int,refresh_rate::Int)
    return LazyUpdate(lazy_block, refresh_rate, 1)
end

function FrankWolfe.select_update_indices(update::LazyUpdate, s::FrankWolfe.CallbackState, dual_gaps)
    #Returns a sequence of randomized cheap indices by 
    #excluding update.lazy_block until "refresh_rate" updates
    #occur, then adds an update of everything while mainting
    #randomized order.
    l = length(s.lmo.lmos)
    return push!([[rand(range(1,l)[1:l .!= update.lazy_block]) for _ in range(1,update.block_size)] for _ in 1:(update.refresh_rate -1)], range(1,l))
end
