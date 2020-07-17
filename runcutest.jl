using CUTEst, DataFrames, Dates, NLPModels, NLPModelsIpopt, Percival, Plots, PrettyTables, SolverBenchmark, SolverTools
pyplot(size=(600,400))

const max_time = 60.0
const atol = 1e-6
const rtol = 1e-6
const ctol = 1e-6
const ftol = 1e-4
const max_v_and_c = 100

# List of Ipopt parameters: C.2 Termination
# https://projects.coin-or.org/Ipopt/browser/stable/3.11/Ipopt/doc/documentation.pdf

function runcutest()
  pnames = sort(CUTEst.select(min_con=1, max_var=max_v_and_c, max_con=max_v_and_c, objtype=2:6,
                              # only_equ_con=true,
                              # custom_filter=p -> p["variables"]["number"] ≥ p["constraints"]["number"]
                              ))
  problems = (CUTEstModel(p) for p in pnames)

  percival_wrapper(nlp; kwargs...) = percival(nlp,
                                              max_time=max_time,
                                              atol=atol, rtol=rtol, ctol=ctol,
                                              kwargs...)
  ipopt_wrapper(nlp; kwargs...) = ipopt(nlp,
                                        max_cpu_time=max_time,
                                        tol=rtol, # Relative tolerance
                                        dual_inf_tol=atol, # Absolute tolerance
                                        constr_viol_tol=ctol, # Absolute tolerance
                                        print_level=0,
                                        x0=nlp.meta.x0,
                                        nlp_scaling_method="none", # No scaling for fairer comparison
                                        acceptable_iter=0, # Do no stop when "acceptable"
                                        kwargs...)

  solvers = Dict(:Percival => percival, :Ipopt => ipopt_wrapper)

  stats = bmark_solvers(solvers, problems)
end

function profile_and_tables(stats)
  # These are the columns in the final report
  cols = [:name, :nvar, :ncon, :status, :objective, :dual_feas, :primal_feas, :elapsed_time, :evals, :success]
  # Use this for override the column header
  hdr_override = Dict(:objective => "\\(f(x)\\)")

  # Don't create this in the stats
  feas = Dict{Symbol,Vector}(s => df[!,:primal_feas] .≤ ctol for (s,df) in stats)
  fmin = min.(stats[:Percival].objective + .!feas[:Percival] * 1e20,
              stats[:Ipopt].objective + .!feas[:Ipopt] * 1e20)

  for (s,df) in stats
    ineq = abs.(df[!,:objective] - fmin) ./ max.(1.0 ,abs.(fmin)) .<= ftol

    df[!,:success] = min.(ineq, feas[s])*1
    df[!,:evals] = 2 * df[!,:neval_grad] + df[!,:neval_obj]
    # Individual tables
    open("$s.tex", "w") do io
      pretty_latex_stats(io, df[!, cols], hdr_override=hdr_override)
    end
  end

  np = size(stats[:Ipopt], 1)

  # For the table with both solvers stacked
  cols = [:name, :nvar, :ncon, :status, :objective, :primal_feas, :evals, :success]
  solvers = collect(keys(stats)) # To set an order
  df_stacked = vcat([stats[s][!, cols] for s in solvers]...)
  # Create a solumn with the name of the solver
  df_stacked[!,:solver] = repeat(string.(solvers), inner=np)
  # Sort by problem name and solver
  df_stacked = sort(df_stacked, [:name, :solver])
  # Remove repeat problems names
  df_stacked[2:2:end,:name] .= ""
  cols = [:name; :solver; cols[2:end-1]] # Change order of name and drop :success

  # The following lines define the highlighting of text and math for successful rows
  # int bold treament in highlighter
  treat_int(x) = begin
    # \(   305\) -> " 3.05"
    m = match(r"\\\((.*)\\\)", x)

    "\\(\\mathbf{$(m[1])}\\)"
  end
  # float bold treament in highlighter
  treat_float(x) = begin
    # \( 3.05\)e\(+01\) -> " 3.05" and "+01"
    m = match(r"\\\((.*)\\\)e\\\((.*)\\\)", x)

    "\\(\\mathbf{$(m[1])e{$(m[2])}}\\)"
  end
  hls = (LatexHighlighter( (data,i,j) -> (data[i,j] isa String || data[i,j] isa Symbol) && df_stacked[i,:success] == 1, ["textbf"] ),
         LatexHighlighter( (data,i,j) -> data[i,j] isa Integer && df_stacked[i,:success] == 1,
                           (_,_,_,x) -> treat_int(x)),
         LatexHighlighter( (data,i,j) -> data[i,j] isa AbstractFloat && df_stacked[i,:success] == 1,
                           (_,_,_,x) -> treat_float(x)))
  # Now create the table
  open("joined.tex", "w") do io
    pretty_latex_stats(io, df_stacked[!, cols], hlines=2:2:size(df_stacked,1), highlighters=hls)
  end

  # Profiles
  # ========
  Plots.default(; titlefontsize=12, legendfontsize=12, tickfontsize=12, guidefontsize=12,
                legend=:bottomright, lw=2, linestyle=:auto)

  # We'll use two sets of profiles: using fmin and usign status
  costsets = [
    (
      [df -> (df.success .!= true) * Inf + df.elapsed_time,
        df -> (df.success .!= true) * Inf + df.evals],
      ["Tempo - \$f_{\\min}\$",
        "Avaliações - \$f_{\\min}\$"],
      "fmin"
    ),
    (
      [df -> (df.status .!= :first_order) * Inf + df.elapsed_time,
        df -> (df.status .!= :first_order) * Inf + df.evals],
      ["Tempo - flag de saída",
        "Avaliações - flag de saída"],
      "status"
    )
  ]

  # We'll use the following subsets of problems
  all_problems = stats[:Ipopt][!,:name]
  cutest_select = Dict(:min_con=>1, :max_var=>max_v_and_c, :max_con=>max_v_and_c, :objtype=>2:6)
  subsets = [
    (
      "todos problemas", # "all problems",
      "all",
      all_problems
    ),
    (
      "igualdades e caixa", #"equalities and bounds",
      "equbnd",
      CUTEst.select(;cutestselect..., only_equ_con=true, only_bnd_var=true) ∩ all_problems
    ),
    (
      "desigualdades e caixa", #"inequalities and bounds",
      "ineqbnd",
      CUTEst.select(;cutestselect..., only_ineq_con=true, only_bnd_var=true) ∩ all_problems
    ),
    (
      "só desigualdades", #"only equalities",
      "equ",
      CUTEst.select(;cutestselect..., only_equ_con=true, only_free_var=true) ∩ all_problems
    ),
    (
      "só igualdades", #"only inequalities",
      "ineq",
      CUTEst.select(;cutestselect..., only_ineq_con=true, only_free_var=true) ∩ all_problems
    ),
    (
      "≤ 10",
      "le10",
      CUTEst.select(min_con=1, max_var=10, max_con=10, objtype=2:6) ∩ all_problems
    ),
    (
      "≤ 100",
      "le100",
      CUTEst.select(min_con=1, max_var=100, max_con=100, objtype=2:6) ∩ all_problems
    ),
    (
      "≥ 100",
      "ge100",
      CUTEst.select(min_var=100, min_con=100, objtype=2:6) ∩ all_problems
    )
  ]

  isdir("profiles") || mkdir("profiles")
  open("images.tex", "w") do io
    println(io, "\\begin{center}")
    for (subsetname, subsetsuffix, subset) in subsets
      I = indexin(subset, all_problems)
      if length(I) == 0
        @warn("Empty subset $subsetname")
        continue
      end
      ss_stats = Dict(s => df[I,:] for (s,df) in stats)
      nss = length(I)
      for (costs, costnames, costsuffix) in costsets
        titles = costnames .* " - $subsetname - $nss problemas"
        p = profile_solvers(ss_stats, costs, titles, width=600, height=500)
        for i = 1:length(costs)
          xlabel!(p[i], "Parâmetro τ")
        end
        ylabel!(p[1], "Proporção de problemas ρ(τ)")
        fname = "profiles/$costsuffix-$subsetsuffix"
        png(fname)
        println(io, "\\includegraphics[width=0.9\\textwidth]{$fname}\n\\vspace{1.0cm}\n")
      end
    end
    println(io, "\\end{center}")
  end
end

isdir("saved-stats") || mkdir("saved-stats")

stats = runcutest()

save_fname = "saved-stats/results-" * Dates.format(now(), "YYYY-mm-ddTHH:MM:SS") * ".jld"
save_stats(stats, save_fname)

# For loading - notice the name below
# load_fname = "saved-stats/results-results-2020-07-16T13:46:02.jld"
# stats2 = load_stats(load_fname)

profile_and_tables(stats)