using Plots
using FiniteDifferences

"""
plot_results
Given a series of list, generate subplots.
list_data_y -> contains a list of a list of lists (where each list refers to a subplot, and a list of lists refers to the y-values of the series inside a subplot).
list_data_x -> contains a list of a list of lists (where each list refers to a subplot, and a list of lists refers to the x-values of the series inside a subplot).
So if we have one plot with two series, these might look like:
    list_data_y = [[[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]]
    list_data_x = [[[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]]
And if we have two plots, each with two series, these might look like:
    list_data_y = [[[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]], [[7, 8, 9, 10, 11, 12], [7, 8, 9, 10, 11, 12]]]
    list_data_x = [[[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]], [[7, 8, 9, 10, 11, 12], [7, 8, 9, 10, 11, 12]]]
list_label -> contains the labels for the series that will be plotted,
which has to have a length equal to the number of series that are being plotted:
    list_label = ["Series 1", "Series 2"]
list_axis_x -> contains the labels for the x-axis that will be plotted,
which has to have a length equal to the number of subplots:
    list_axis_x = ["x-axis plot 1", "x-axis plot 1"]
list_axis_y -> Same as list_axis_x but for the y-axis
xscalelog -> A list of values indicating the type of axes to use in each subplot,
must be equal to the number of subplots:
    xscalelog = [:log, :identity]
yscalelog -> Same as xscalelog but for the y-axis
"""
function plot_results(
    list_data_y,
    list_data_x,
    list_label,
    list_axis_x,
    list_axis_y;
    filename=nothing,
    xscalelog=nothing,
    yscalelog=nothing,
    legend_position=nothing,
    list_style=fill(:solid, length(list_label)),
    list_color=get_color_palette(:auto, plot_color(:white)),
    list_markers=[
        :circle,
        :rect,
        :utriangle,
        :diamond,
        :hexagon,
        :+,
        :x,
        :star5,
        :cross,
        :xcross,
        :dtriangle,
        :rtriangle,
        :ltriangle,
        :pentagon,
        :heptagon,
        :octagon,
        :star4,
        :star6,
        :star7,
        :star8,
        :vline,
        :hline,
    ],
    number_markers_per_line=10,
    line_width=3.0,
    marker_size=5.0,
    transparency_markers=0.45,
    font_size_axis=12,
    font_size_legend=9,
)
    gr()
    plt = nothing
    list_plots = Plots.Plot{Plots.GRBackend}[]
    #Plot an appropiate number of plots
    for i in eachindex(list_data_x)
        for j in eachindex(list_data_x[i])
            if isnothing(xscalelog)
                xscale = :identity
            else
                xscale = xscalelog[i]
            end
            if isnothing(yscalelog)
                yscale = :log
            else
                yscale = yscalelog[i]
            end
            if isnothing(legend_position)
                position_legend = :best
                legend_display = true
            else
                position_legend = legend_position[i]
                if isnothing(position_legend)
                    legend_display = false
                else
                    legend_display = true
                end
            end
            if j == 1
                if legend_display
                    plt = plot(
                        list_data_x[i][j],
                        list_data_y[i][j],
                        label="",
                        xaxis=xscale,
                        yaxis=yscale,
                        ylabel=list_axis_y[i],
                        xlabel=list_axis_x[i],
                        legend=position_legend,
                        yguidefontsize=font_size_axis,
                        xguidefontsize=font_size_axis,
                        legendfontsize=font_size_legend,
                        width=line_width,
                        linestyle=list_style[j],
                        color=list_color[j],
                        grid=true,
                    )
                else
                    plt = plot(
                        list_data_x[i][j],
                        list_data_y[i][j],
                        label="",
                        xaxis=xscale,
                        yaxis=yscale,
                        ylabel=list_axis_y[i],
                        xlabel=list_axis_x[i],
                        yguidefontsize=font_size_axis,
                        xguidefontsize=font_size_axis,
                        width=line_width,
                        linestyle=list_style[j],
                        color=list_color[j],
                        grid=true,
                    )
                end
            else
                if legend_display
                    plot!(
                        list_data_x[i][j],
                        list_data_y[i][j],
                        label="",
                        width=line_width,
                        linestyle=list_style[j],
                        color=list_color[j],
                        legend=position_legend,
                    )
                else
                    plot!(
                        list_data_x[i][j],
                        list_data_y[i][j],
                        label="",
                        width=line_width,
                        linestyle=list_style[j],
                        color=list_color[j],
                    )
                end
            end
            if xscale == :log
                indices =
                    round.(
                        Int,
                        10 .^ (range(
                            log10(1),
                            log10(length(list_data_x[i][j])),
                            length=number_markers_per_line,
                        )),
                    )
                scatter!(
                    list_data_x[i][j][indices],
                    list_data_y[i][j][indices],
                    markershape=list_markers[j],
                    markercolor=list_color[j],
                    markersize=marker_size,
                    markeralpha=transparency_markers,
                    label=list_label[j],
                    legend=position_legend,
                )
            else
                scatter!(
                    view(
                        list_data_x[i][j],
                        1:length(list_data_x[i][j])÷number_markers_per_line:length(
                            list_data_x[i][j],
                        ),
                    ),
                    view(
                        list_data_y[i][j],
                        1:length(list_data_y[i][j])÷number_markers_per_line:length(
                            list_data_y[i][j],
                        ),
                    ),
                    markershape=list_markers[j],
                    markercolor=list_color[j],
                    markersize=marker_size,
                    markeralpha=transparency_markers,
                    label=list_label[j],
                    legend=position_legend,
                )
            end
        end
        push!(list_plots, plt)
    end
    fp = plot(list_plots..., layout=length(list_plots))
    plot!(size=(600, 400))
    if filename !== nothing
        savefig(fp, filename)
    end
    return fp
end

# Recipe for plotting markers in plot_trajectories
@recipe function f(::Type{Val{:samplemarkers}}, x, y, z; n_markers=10, log=false)
    n = length(y)

    # Choose datapoints for markers
    if log
        xmin = log10(x[1])
        xmax = log10(x[end])
        thresholds = collect(xmin:(xmax-xmin)/(n_markers-1):xmax)
        indices = [argmin(i -> abs(t - log10(x[i])), eachindex(x)) for t in thresholds]
    else
        indices = 1:Int(ceil(length(x) / n_markers)):n
    end
    sx, sy = x[indices], y[indices]

    # add an empty series with the correct type for legend markers
    @series begin
        seriestype := :path
        markershape --> :auto
        x := []
        y := []
    end
    # add a series for the line
    @series begin
        primary := false # no legend entry
        markershape := :none # ensure no markers
        seriestype := :path
        seriescolor := get(plotattributes, :seriescolor, :auto)
        x := x
        y := y
    end
    # return  a series for the sampled markers
    primary := false
    seriestype := :scatter
    markershape --> :auto
    x := sx
    y := sy
    z_order := 1
end

function plot_trajectories(
    data,
    label;
    filename=nothing,
    xscalelog=false,
    yscalelog=true,
    dual_averaging=true,
    dual_min = false,
    legend_position=:topright,
    lstyle=fill(:solid, length(data)),
    marker_shapes=nothing,
    n_markers=10,
    reduce_size=false,
    primal_offset=1e-8,
    line_width=1.3,
    empty_marker=false,
)
    # theme(:dark)
    # theme(:vibrant)
    Plots.gr()

    x = []
    y = []
    offset = 2

    function sub_plot(idx_x, idx_y; legend=false, xlabel="", ylabel="", y_offset=0)

        fig = nothing

        for (i, trajectory) in enumerate(data)

            l = length(trajectory)
            if reduce_size && l > 1000
                indices = Int.(round.(collect(1:l/1000:l)))
                trajectory = trajectory[indices]
            end

            x = [trajectory[j][idx_x] for j in offset:length(trajectory)]
            if dual_averaging && (idx_y == 4)
                csum = 0
                y = zeros(length(trajectory)-offset+1)
                for j in range(offset,length(trajectory))
                    csum += trajectory[j][idx_y] + y_offset
                    y[j-offset+1] = csum./(j-offset+1)
                end
            elseif dual_min && (idx_y == 4)
                y = zeros(length(trajectory)-offset+1)
                y[1] = trajectory[offset][idx_y]
                for j in range(offset+1,length(trajectory))
                    y[j-offset+1] = min(y[j-offset], trajectory[j][idx_y] + y_offset)
                end
            else
            y = [trajectory[j][idx_y] + y_offset for j in offset:length(trajectory)]
            end

            if marker_shapes !== nothing && n_markers >= 2
                marker_args = Dict(
                    :st => :samplemarkers,
                    :n_markers => n_markers,
                    :shape => marker_shapes[i],
                    :log => xscalelog,
                    :markercolor => empty_marker ? :white : :match,
                    :markerstrokecolor => empty_marker ? i : :match,
                )
            else
                marker_args = Dict()
            end

            if i == 1
                fig = plot(
                    x,
                    y,
                    label=label[i],
                    xaxis=xscalelog ? :log : :identity,
                    yaxis=yscalelog ? :log : :identity,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    legend=legend,
                    yguidefontsize=8,
                    xguidefontsize=8,
                    legendfontsize=8,
                    width=line_width,
                    linestyle=lstyle[i];
                    marker_args...,
                )
            else
                plot!(x, y, label=label[i], width=line_width, linestyle=lstyle[i]; marker_args...)
            end
        end
        return fig
    end

    pit = sub_plot(1, 2; legend=legend_position, ylabel="Primal", y_offset=primal_offset)
    pti = sub_plot(5, 2; y_offset=primal_offset)
    if dual_averaging
        dit = sub_plot(1, 4; xlabel="Iterations", ylabel="Avg FW gap")
    elseif dual_min
        dit = sub_plot(1, 4; xlabel="Iterations", ylabel="Min FW gap")
    else
        dit = sub_plot(1, 4; xlabel="Iterations", ylabel="FW gap")
    end
    

    dti = sub_plot(5, 4; xlabel="Time (s)")

    fp = plot(pit, pti, dit, dti, layout=(2, 2)) # layout = @layout([A{0.01h}; [B C; D E]]))
    plot!(size=(600, 400))
    if filename !== nothing
        savefig(fp, filename)
    end
    return fp
end

function export_data(
        """
        Outputs data in a latex-readable format.
        """
    data_trajectories,
    label;
    filename_prefix="FW_results",
    filename_suffix="",
    iter_skip=100,
    compute_FWgaps=true,
    )
    for L in range(1,length(data_trajectories))
        data = data_trajectories[L]
        filename = string(filename_prefix,label[L],filename_suffix,".txt")
        open(filename,"a") do io
            #Headers for txt output file readable by latex
            #If FW gaps were default from FW.jl, they are
            #not actual gaps; reflect the data file naming
            #scheme, so we don't plot something incorrect.
	    if compute_FWgaps == false
                println(io,"iter time primal fd fdmin fdavg, lmo1 lmo2")
            else
                #For reporting purposes, still record the approximation of
                #the F-W gap recorded by FW.jl
                println(io,"iter time primal d dmin davg lmo1 lmo2")
            end
            #data looks like:
#                    t,
#                    primal,
#                    primal - dual_gap,
#                    dual_gap,
#                    tot_time,
            csum = 0
            dual_min = Inf.*ones(length(data))
            dual_min[1] = data[1][4]
            dual_avg = zeros(length(data))
            for j in range(1,length(data))
               csum += data[j][4] 
               dual_avg[j] = csum./j
               if j>=2
                   dual_min[j] = min(dual_min[j-1], data[j][4])
               end
            end
            for i in range(1,step=iter_skip,stop=length(data))
		    println(io, data[i][1]," ", data[i][5]," ", data[i][2]," ", data[i][4]," ", dual_min[i]," ", dual_avg[i]," ",data[i][6]," ",data[i][7])
            end
        end
    end
end

function plot_sparsity(
    data,
    label;
    filename=nothing,
    xscalelog=false,
    legend_position=:topright,
    yscalelog=true,
    lstyle=fill(:solid, length(data)),
    marker_shapes=nothing,
    n_markers=10,
    empty_marker=false,
    reduce_size=false,
)
    Plots.gr()

    xscale = xscalelog ? :log : :identity
    yscale = yscalelog ? :log : :identity
    offset = 2

    function subplot(idx_x, idx_y, ylabel)

        fig = nothing
        for (i, trajectory) in enumerate(data)

            l = length(trajectory)
            if reduce_size && l > 1000
                indices = Int.(round.(collect(1:l/1000:l)))
                trajectory = trajectory[indices]
            end


            x = [trajectory[j][idx_x] for j in offset:length(trajectory)]
            y = [trajectory[j][idx_y] for j in offset:length(trajectory)]
            if marker_shapes !== nothing && n_markers >= 2
                marker_args = Dict(
                    :st => :samplemarkers,
                    :n_markers => n_markers,
                    :shape => marker_shapes[i],
                    :log => xscalelog,
                    :startmark => 5 + 20 * (i - 1),
                    :markercolor => empty_marker ? :white : :match,
                    :markerstrokecolor => empty_marker ? i : :match,
                )
            else
                marker_args = Dict()
            end
            if i == 1
                fig = plot(
                    x,
                    y;
                    label=label[i],
                    xaxis=xscale,
                    yaxis=yscale,
                    ylabel=ylabel,
                    legend=legend_position,
                    yguidefontsize=8,
                    xguidefontsize=8,
                    legendfontsize=8,
                    linestyle=lstyle[i],
                    marker_args...,
                )
            else
                plot!(x, y; label=label[i], linestyle=lstyle[i], marker_args...)
            end
        end

        return fig
    end

    ps = subplot(6, 2, "Primal")
    ds = subplot(6, 4, "FW gap")

    fp = plot(ps, ds, layout=(1, 2)) # layout = @layout([A{0.01h}; [B C; D E]]))
    plot!(size=(600, 200))
    if filename !== nothing
        savefig(fp, filename)
    end
    return fp
end

"""
Check if the gradient using finite differences matches the grad! provided.
"""
function check_gradients(grad!, f, gradient, num_tests=10, tolerance=1.0e-5)
    for i in 1:num_tests
        random_point = similar(gradient)
        random_point .= rand(length(gradient))
        grad!(gradient, random_point)
        if norm(grad(central_fdm(5, 1), f, random_point)[1] - gradient) > tolerance
            @warn "There is a noticeable difference between the gradient provided and
            the gradient computed using finite differences.:\n$(norm(grad(central_fdm(5, 1), f, random_point)[1] - gradient))"
        end
    end
end