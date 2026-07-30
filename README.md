This repository contains the code used to produce the results in **"Flexible block-iterative analysis for the Frank-Wolfe algorithm"**
by Gábor Braun, Jannis Halbey, Sebastian Pokutta, and Zev Woodstock (<a href="https://zevwoodstock.github.io/media/publications/block.pdf">preprint</a>)

After cloning the repository, to run an experiment, execute the following code (make sure <a href="https://julialang.org/">Julia</a> version $\geq 1.8.5$ is installed beforehand):

`julia --project`

Then, to run Experiment 1 in the article (the convex problem) for both BCFW with short-step and adaptive step sizes:

`julia> include("test.jl")`

`julia> include("test_adaptive.jl")`

Similarly, to run Experiment 2 in the article (the nonconvex Difference-of-Convex "DC" problem):

`julia> include("dc-test.jl")`

After each experiment completes, they will output LaTeX-readable `.txt` files of the form

`[header]_A[number_of_Averaged_trials]_[n value]_[activation method]-[timestamp].txt`,
where:
- `[header]` is `BCFW` and `BCFW_adpative` for Experiment 1 with short step and adaptive step size respectively, and `DC` for Experiment 2.
- `[number_of_Averaged_trials]` is given by the variable `num_trials` in each test script.
- `[n value]` is the length of the side of the matrix variable for each problem (i.e., each problem has $n^2$ variables). The list of all `n` values considered in an experiment is prescribed in the `n_list` variable of each test script.
- `[activation method]` is one of the following:
  - `full`: corresponding to full activation;
  - `cyclic`: corresponding to cyclic activation;
  - `stoc`: corresponding to permuted-cyclic activation;
  - `ecycN`: corresponding to lazified essentially-cyclic singleton activation where the expensive LMO is activated every `N` iterations;
  - `customN`: corresponding to a lazy method where a full activation is only performed once every `N` iterations.

The specific data from our experiments is available in the `results` directory.

Each column of a results file has a corresponding label:
- `iter`: Iteration
- `time`: Time used
- `primal`: Function value at the current iteration
   `lmo1` / `lmo2` - number of linear minimization oracle calls completed until the current iteration (see experiment scripts for their corresponding constraint set).
- `d` (or `fd`): This label is `d` if it is the true Frank-Wolfe gap value at the current iteration; otherwise, an approximate (albeit incorrect) value is given by `fd`. Since BCFW does not compute full Frank-Wolfe gaps on the fly (only partial F-W gaps are available), this statistic must be computed separately, hence increasing experiment runtime. By default, full Frank-Wolfe gaps are only computed on the nonconvex DC runs; to compute full Frank-Wolfe gaps on any experiment, set `compute_FWgaps=true` within the experiment script.
<!---
More specifically, `d` corresponds to (in the notation of the article) 
$$\sum_{i\in I} \langle \nabla f(\boldsymbol{x}^i_t)\,|\,\boldsymbol{v}^i_t-\boldsymbol{x}^i_t\rangle$$,
while `fd` is just a placeholder value; specifically, it is
$\sum_{i\in I}\langle \nabla f(\boldsymbol{x}^i_{c_i(t)}\,|\,\boldsymbol{v}^i_{c_i(t)}-\boldsymbol{x}^i_{c_i(t)}\rangle$, where $c_i(t)$ is the most recent iteration preceding $t$ at which component $i$ was updated. The values of `fd` are never plotted in the article.
--->
- `dmin` (or `fdmin`) - minimally observed Frank-Wolfe gap (resp. F-W gap approximation) observed until the current iteration
- `davg` (or `favg`) - average of all Frank-Wolfe gaps (resp. F-W gap approximations) observed until the current iteration
  
To plot the data, one can use, e.g., TikZ/pgfplots. The TeX code for plotting all of our data will be available on ArXiv, but a brief example (plotting Experiment 1 for the averaged $n=500$ results) is below (please note the use of custom colors for accessibility reasons; see <a href="https://github.com/jfdm/sta-latex/blob/master/colour-blind.sty">this repository</a> for their definitions)

```
\begin{tikzpicture}[scale=1.0]
\begin{axis}[height=4.2cm,width=4.1cm, legend cell align={left},
ylabel style={yshift=-0.75em,font=\scriptsize}, 
xlabel style={yshift=0.50em,font=\scriptsize},
tick label style={font=\tiny},
legend columns=2,
legend
entries={Full,$5$-Lazy,Cyclic,$10$-Lazy,P-Cyclic,$20$-Lazy,E-Cyclic},
legend style={
    font=\scriptsize,
    at={(1.0,1.75)},
    anchor=north east,
    },
ytick={1e-1,1,1e1},
%scaled x ticks=false,
%xtick={0,2500,5000,7500,10000},
xlabel=Time (sec),
%ylabel=$f(x)-f(x^*)$,
grid,
xmin =0, xmax=300, ymin=5e-2, ymax=1e2, ymode=log, mark repeat=18]
\addplot+[thick, mark=triangle, mark repeat=15, color=cb-burgundy] table[x={time}, y={primal}] {results/BCFW_A20_500_full-07-18T19-41.txt};
\addplot+[thick, mark=oplus, color=cb-brown]
table[x={time}, y={primal}] {results/BCFW_A20_500_custom5-07-18T19-41.txt};
\addplot+[thick, mark=square, color=cb-blue] table[x={time}, y={primal}] {results/BCFW_A20_500_cyclic-07-18T19-41.txt};
\addplot+[thick, mark=star, color=cb-lilac]
table[x={time}, y={primal}] {results/BCFW_A20_500_custom10-07-18T19-41.txt};
\addplot+[thick, mark=o, color=cb-green-sea] table[x={time}, y={primal}] {results/BCFW_A20_500_stoc-07-18T19-41.txt};
\addplot+[thick,solid, mark=diamond, color=cb-salmon-pink]
table[x={time}, y={primal}] {results/BCFW_A20_500_custom20-07-18T19-41.txt};
\addplot+[thick,solid, mark=otimes, mark repeat=30, color=cb-blue-light]
table[x={time}, y={primal}] {results/BCFW_A20_500_ecyc20-07-18T19-41.txt};
\end{axis}
\end{tikzpicture}

```
The work of ZW was supported by NSF DMS-2532423.
