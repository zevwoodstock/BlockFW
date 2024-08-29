using FrankWolfe
using LinearAlgebra
using Dates
using Random

include("src/bcg.jl")
include("plot_utils.jl")

#All data from the current run will have a suffix
#"-MM-DDTHH-MM.txt" and be in a latex-readable format
datetime = string(now())
filename_suffix = string(datetime[5:13],"-",datetime[15:16])


#Idea: Make an objective function f(x1,x2) = dot(x, Ax) -
#dot(x2,Bx2), where x is in the l1 ball (lots of separable
#constraints), and x2 in nuclear norm ball (LMO advantaged,
#expensive). Form A and B such that A-B is INDEFINITE. Then it's
#neither convex nor concave, and L_f=||A-B||, is calculable.

n_list = [100, 300, 500]
num_trials = 20
#Set compute_FWgaps = true for this case, since we're solving a
#nonconvex problem, this needed to establish stationarity.
compute_FWgaps = true

for en in n_list

global n = en
println("Starting n = ",n)
global nn = n*n
global n2 = 2*n
#Setup LMO initializations
global lmo1 = FrankWolfe.TrackingLMO(FrankWolfe.ScaledBoundLInfNormBall(-ones(n), ones(n)))
global lmo2 = FrankWolfe.TrackingLMO(FrankWolfe.NuclearNormLMO(1.0))
lmo_list = Vector{FrankWolfe.LinearMinimizationOracle}()
append!(lmo_list, [lmo1 for _ in range(1,n)])
push!(lmo_list, lmo2)
prod_lmo = FrankWolfe.ProductLMO{n+1, Vector{FrankWolfe.LinearMinimizationOracle}}(lmo_list)
#Set up custom activation schemes
lazy_component = n+1
lazy_skiprate = 2
lazy_blocksize = n
lazy_skiprate2 = 5
lazy_blocksize2 = round(n/2)
lazy_skiprate3 = 10
lazy_blocksize3 = 10
lazy_skiprate3 = 10
lazy_blocksize3 = 10
lazy_skiprate4 = 20
lazy_blocksize4 = 2
#Set up orders to be tested using FW.jl
orders = [
    FrankWolfe.FullUpdate(),
    FrankWolfe.CyclicUpdate(),
    FrankWolfe.StochasticUpdate(),
    LazyUpdate(lazy_component,lazy_skiprate, lazy_blocksize),
    LazyUpdate(lazy_component,lazy_skiprate2, lazy_blocksize2),
    LazyUpdate(lazy_component,lazy_skiprate3, lazy_blocksize3),
    LazyUpdate(lazy_component,lazy_skiprate4, lazy_blocksize4),
]
#Compute a maximum number of iterations for each algorithm. FW.jl
#counts an "iteration" as an external iteration, so we adjust the
#maximum number of iterations accordingly (e.g., for a problem with
#'m' constraints, FW.jl counts m iterations of cyclic as 1
#iteration. So, here we adjust our maximum iteration counter so
#that everyone actually does the same number of iterations.
maxiter_full = 10000
#This is the variable for "FW.jl iterations" (External iterations)
max_iters = (maxiter_full,
	     convert(Int,ceil(maxiter_full/length(prod_lmo.lmos))),
	     convert(Int,ceil(maxiter_full/length(prod_lmo.lmos))),
	     convert(Int,ceil(maxiter_full/lazy_skiprate)),
	     convert(Int,ceil(maxiter_full/lazy_skiprate2)),
	     convert(Int,ceil(maxiter_full/lazy_skiprate3)),
	     convert(Int,ceil(maxiter_full/lazy_skiprate4)),
	     )
#Use these multipliers on the output of FW.jl to compute the number
#actual iterations taken.
iter_multiplier = [1,
		   length(prod_lmo.lmos),
		   length(prod_lmo.lmos),
		   lazy_skiprate,
		   lazy_skiprate2,
		   lazy_skiprate3,
		   lazy_skiprate4,
		   ]
#Labels for files output using export_data() in plot_utils.j.
#Provide a string for each element of orders.
labels_filename = ("full", 
		     "cyclic", 
		     "stoc", 
		     string("custom",lazy_skiprate),
                     string("custom",lazy_skiprate2),
                     string("custom",lazy_skiprate3),
		     string("custom",lazy_skiprate4),
		     )


function project_psd(A)
    #symmetricize
    A = 0.5*(A'+A)
    #project
    e, U = eigen(A)
    e = real.(e)
    e = (e.>=0).*e
    return U*Diagonal(e)*(U')
end

#Initialize variable for holding averaged statistics.
gaps = [[] for i in range(1,length(orders))]

for i in range(1,length(orders))
    #Array to hold information for each trial; will use for
    #averaging at the end.  iters, f(x), f(x)-gap, gap, time, lmo1
    #count, lmo2 count
    temp_trialdata = zeros(max_iters[i]+3,7,num_trials)
    for trial in range(1,num_trials)
        println("Starting trial ",trial)
        trajectory= []
        xs = []
        #Set seed to trial number for reproducibility.
        Random.seed!(trial)
        #Generate problem
        global AA = project_psd(randn(n2,n2))
        global BB = project_psd(randn(n2,n2))
        #Create a kernel which is the difference of two convex
        #functions
        global CC = AA - BB
        
        #Confirm that matrix is indefinite:
	v = real.(eigen(CC).values)
        if maximum(v)> 0 && minimum(v) < 0
            println("Confirmed: Problem is indefinite.")
        else
            println("Problem is either positive or negative semidefinite.")
        end 
        
        function f(x)
        	x_mat = hcat(x.blocks[1:n+1]...)
            return 0.5*(FrankWolfe.fast_dot(x_mat,x_mat*CC))
        end
        
        function grad!(storage, x)
        #    n = size(x.blocks[1])[1]
        #    nn = n*n
            Ax_mat = hcat(x.blocks[1:n+1]...)*CC
            storage.blocks[1:n] = [Ax_mat[:,i] for i in range(1,n)]
            storage.blocks[n+1] = Ax_mat[:,n+1:n2]
        end
        
        x0_vec = Vector{Matrix{Float64}}([randn(n,1) .- 0.5 for _ in range(1,n)])
        push!(x0_vec,randn(n,n))
        x0 = FrankWolfe.compute_extreme_point(prod_lmo, FrankWolfe.BlockVector(x0_vec) )
        #reset LMO counter for each trial, don't include initialization
        lmo1.counter = 0
        lmo2.counter = 0

    
        #Variables to approximate timing of first iteration of FW.jl.
        temp_vals = []
        init_time = []
        #Callbacks to be used within FW.jl to record statistics as
        #FW.jl runs; this is necessary for recording statistics from
        #partial runs where FW.jl hits a snag (e.g., for Arpack.jl
        #instabilities).
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
        try
        xf, _, _, _, _ = FrankWolfe.block_coordinate_frank_wolfe(
            f,
            grad!,
            prod_lmo,
            x0;
            verbose=true,
            trajectory=false,
            update_order=orders[i],
            callback=mycallback,
            max_iteration=max_iters[i],
            line_search=FrankWolfe.Shortstep(norm(CC)),
            epsilon=0,
        )
        catch e
        println("Error occurred during FW run.")
        end
    
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
    export_data((gaps[i],), (labels_filename[i],), filename_prefix=string("DC_A",num_trials,"_",n,"_"),filename_suffix=filename_suffix,compute_FWgaps=compute_FWgaps,iter_skip=convert(Int,ceil(max_iters[i]/100)))
    println("Done.")
end
end


