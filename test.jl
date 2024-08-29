using FrankWolfe
using LinearAlgebra
using Dates
using Random

#Gather information for logging...
datetime = string(now())
filename_suffix = string(datetime[5:13],"-",datetime[15:16])

include("src/BlockFW.jl")
include("plot_utils.jl")
#Quadratic loss
f(x) = dot(x.blocks[1][:] - x.blocks[2], x.blocks[1][:] - x.blocks[2])

function grad!(storage, x)
    n = size(x.blocks[1])[1]
    g = copy(x)
    g.blocks = [x.blocks[1] - reshape(x.blocks[2],n,n), x.blocks[2] - x.blocks[1][:]]
    @. storage = g
end

#List of values of n (problems defined on n-by-n matrix variables)
n_list = [100, 300, 500]
num_trials = 20

for n in n_list

println("Starting n = ", n)
n2 = n*n
#Since complete Frank-Wolfe gaps are not computed (unless full
#activation is done) on-the-fly within the algorithm, this Boolean
#is used to tell the algorithm to compute the full gaps for
#reporting purposes. Warning: this can cause slow-downs.
compute_FWgaps = false
#Define problem parameters as in Experiment 1
global lmo1 = FrankWolfe.TrackingLMO(FrankWolfe.SpectraplexLMO(1,n))
global lmo2 = FrankWolfe.TrackingLMO(FrankWolfe.ScaledBoundLInfNormBall(-ones(n2).*5, ones(n2)*(1/n)))
prod_lmo = FrankWolfe.ProductLMO((lmo1, lmo2))
#Various values of p-lazy to study:
lazy_component = 1
lazy_skiprate = 20
lazy_skiprate2 = 10
lazy_skiprate3 = 5
#Define a list of block-selection strategies to iterate through
orders = [
    FrankWolfe.FullUpdate(),
    FrankWolfe.CyclicUpdate(),
    FrankWolfe.StochasticUpdate(),
    LazyUpdate(lazy_component,lazy_skiprate),
    LazyUpdate(lazy_component,lazy_skiprate2),
    LazyUpdate(lazy_component,lazy_skiprate3),
]
#Since FrankWolfe.jl counts an "iteration" as a full external
#iteration, we need to use these multipliers to keep iteration
#counting consistent.
iter_multiplier = [1,
		   length(prod_lmo.lmos),
		   length(prod_lmo.lmos),
		   lazy_skiprate,
		   lazy_skiprate2,
		   lazy_skiprate3,
		   ]
labels_filename = ("full", 
		     "cyclic", 
		     "stoc", 
		     string("custom",lazy_skiprate),
		     string("custom",lazy_skiprate2),
		     string("custom",lazy_skiprate3),
		     )
#FW.jl currently counts a full "epoch" as an "iteration", so to get the actual
#iteration counting to work out properly, we need to do (1/ number of
#constraints) FW.jl iterations for cyclic and stochastic, then
#do 1/lazy_skiprate FW.jl iterations for the custom method.
maxiter_full = 10000
max_iters = (maxiter_full,
	     convert(Int,round(0.5*maxiter_full)),
             convert(Int,round(0.5*maxiter_full)),
             convert(Int,round(maxiter_full/lazy_skiprate)),
             convert(Int,round(maxiter_full/lazy_skiprate2)),
             convert(Int,round(maxiter_full/lazy_skiprate3)),
	     )

#Initialize
gaps = [[] for i in range(1,length(orders))]

for i in range(1,length(orders))
     #Array to hold information for each trial; will use this for
     #averaging at the end.
     temp_trialdata = zeros(max_iters[i]+3,7,num_trials)
     for trial in range(1,num_trials)
     println("Starting trial ",trial)
     trajectory= []
     xs = []
     #Set seed to trial number for reproducibility.
     Random.seed!(trial)
     x0 = FrankWolfe.compute_extreme_point(prod_lmo,FrankWolfe.BlockVector([randn(n,n), randn(n2).-0.5]))
     #reset LMO counter, do not count initialization
     lmo1.counter = 0
     lmo2.counter = 0
     #Keeping time in FW.jl doesn't count the first iteration, so we
     #estimate it; initializing variables here:
     temp_vals = []
     init_time = []
     #Define two callbacks (used for recording statistics)
     #post-run. They only differ in whether or not they re-compute
     #the Frank-Wolfe gap.
     function traj_mycallback_x(state,args...)
        if state.t==1
            #Copy initial values in as well.
            push!(trajectory, (0, f(x0), 0, 0, 0, 0, 0, x0))
            #Temporarily record data for first iteration; there is
            #an issue with FW.jl which does not record the time for
            #the first external iteration; hence, we are going to
            #approximate it with the amount of time the 2nd
            #iteration takes (by default, FW.jl just reports this
            #time as zero).
            temp_vals = (copy(state.t), copy(state.primal), copy(state.dual), copy(state.dual_gap), copy(state.time), copy(lmo1.counter), copy(lmo2.counter), copy(state.x))
        elseif state.t==2
            init_time = copy(state.time)
            push!(trajectory, (temp_vals[1], temp_vals[2], temp_vals[3], temp_vals[4], init_time, temp_vals[6], temp_vals[7], temp_vals[8]))
            push!(trajectory, (copy(state.t), copy(state.primal), copy(state.dual), copy(state.dual_gap), init_time + copy(state.time),  copy(lmo1.counter), copy(lmo2.counter), copy(state.x)))
        else
            push!(trajectory, (copy(state.t), copy(state.primal), copy(state.dual), copy(state.dual_gap), init_time + copy(state.time),  copy(lmo1.counter), copy(lmo2.counter), copy(state.x)))
        end
    end
    function traj_callback(state,args...)
        if state.t==1
            #Copy initial values in as well.
            push!(trajectory, (0, f(x0), 0, 0, 0, 0, 0))
            #Temporarily record data for first iteration; there is
            #an issue with FW.jl which does not record the time for
            #the first external iteration; hence, we are going to
            #approximate it with the amount of time the 2nd
            #iteration takes (by default, FW.jl just reports this
            #time as zero).
            temp_vals = (copy(state.t), copy(state.primal), copy(state.dual), copy(state.dual_gap), copy(state.time), copy(lmo1.counter), copy(lmo2.counter))
        elseif state.t==2
            init_time = copy(state.time)
            push!(trajectory, (temp_vals[1], temp_vals[2], temp_vals[3], temp_vals[4], init_time, temp_vals[6], temp_vals[7]))
            push!(trajectory, (copy(state.t), copy(state.primal), copy(state.dual), copy(state.dual_gap), init_time + copy(state.time), copy(lmo1.counter), copy(lmo2.counter)))
        else
            push!(trajectory, (copy(state.t), copy(state.primal), copy(state.dual), copy(state.dual_gap), init_time + copy(state.time), copy(lmo1.counter), copy(lmo2.counter)))
        end
    end
    mycallback = compute_FWgaps ? traj_mycallback_x : traj_callback
    #Using a try-catch loop; occasionally numerical instabilities
    #cause issues within some of FrankWolfe.jl's dependencies
    #(e.g., Arpack.jl is no longer maintained); this way, we don't
    #lose all our data if we hit a bug in the middle of one run.
    try
    xf, _, _, _, _ = FrankWolfe.block_coordinate_frank_wolfe(
        f,
        grad!,
        prod_lmo,
        x0;
        verbose=false,
        trajectory=false,
        update_order=orders[i],
        callback=mycallback,
        max_iteration=max_iters[i],
        line_search=FrankWolfe.Shortstep(2),
    )
    catch e
    println("Error occurred during FW run.")
    end
    #initialize Gradient storage
    g_storage = similar(x0)
    println("Run complete. Recomputing statistics for reporting...")
    for j in range(1,length(trajectory))
        if  compute_FWgaps
            #This is already a block vector
            Xt = trajectory[j][8] 
            grad!(g_storage,Xt)
            v = FrankWolfe.compute_extreme_point(prod_lmo,g_storage)
            #compute the actual full FW gap, and Count iterations
            #correctly, depending on the update rule selected.
	    temp_trialdata[j,:,trial] = [iter_multiplier[i].*trajectory[j][1], trajectory[j][2], trajectory[j][3], FrankWolfe.fast_dot(g_storage,Xt - v), trajectory[j][5],trajectory[j][6],trajectory[j][7]]
        else
		#If don't care about FW gaps, at least count iteration correctly.
            temp_trialdata[j,:,trial]  = [iter_multiplier[i].*(trajectory[j][1]), trajectory[j][2], trajectory[j][3], trajectory[j][4], trajectory[j][5], trajectory[j][6], trajectory[j][7]]
        end
    end
    print("Done.")
    end #run next trial 
    println("Computing average...")
    #Compute average over all trials for which data was reported:
    #For every iteration number, count the number of trials which actually got to that number of iterations 
    #(termination may occur early if FW.jl determines machine epsilon optimality, or if Arpack.jl is instable); 
    #Use this number in computing averages (all other entries in the sum will be zero).
    num_completed_trials_at_iter = num_trials .- [length(findall(iszero,temp_trialdata[iter,1,:])) for iter in range(1,max_iters[i]+3)]
    #Special handling for iteration 0 (since we use the iteration counter being written above to compute our count).
    num_completed_trials_at_iter[1] = num_trials
    averaged_data = sum(temp_trialdata,dims=3) ./ num_completed_trials_at_iter
    #Package data into tuples for export...
    for j in range(1,max_iters[i]+3)
        iter_data = (averaged_data[j,1], averaged_data[j,2], averaged_data[j,3], averaged_data[j,4], averaged_data[j,5], averaged_data[j,6], averaged_data[j,7])
        push!(gaps[i], iter_data)
    end

    #Write data within the script, so that if something causes a fault we still get correctly-labeled data.
    print("Writing statistics...")
    export_data((gaps[i],), (labels_filename[i],), filename_prefix=string("BCFW_A",num_trials,"_",n,"_"),filename_suffix=filename_suffix,compute_FWgaps=compute_FWgaps,iter_skip=convert(Int,ceil(max_iters[i]/100)))
    println("Done.")
end

end

