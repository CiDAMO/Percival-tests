using CUTEst, NLPModels, NLPModelsIpopt, Percival, Plots, SolverBenchmark, SolverTools
plotlyjs(size=(600,400))

function runcutest()
  pnames = sort(CUTEst.select(min_con=1,
                              max_var=10,
                              max_con=10))
  problems = (CUTEstModel(p) for p in pnames)

  ipopt_wrapper(nlp; kwargs...) = ipopt(nlp,
                                        max_cpu_time=30.0,
                                        print_level=0;
                                        kwargs...)

  solvers = Dict(:Percival => percival, :Ipopt => ipopt_wrapper)

  stats = bmark_solvers(solvers, problems)
end

function profile_and_tables(stats)
  cols = [:name, :nvar, :ncon, :status, :objective, :dual_feas, :primal_feas, :elapsed_time, :neval_obj, :neval_grad, :neval_cons, :neval_jac, :neval_jprod, :neval_jtprod, :neval_hess, :neval_hprod]
  hdr_override = Dict(:objective => "\\(f(x)\\)", :dual_feas => "dual", :primal_feas => "prim", :elapsed_time => "\\(\\Delta t\\)",
                      :neval_obj => "\\#f", :neval_grad => "\\#g", :neval_hess => "\\#H", :neval_hprod => "\\#Hv",
                      :neval_cons => "\\#c", :neval_jac => "\\#J", :neval_jprod => "\\#Jv", :neval_jtprod => "\\#Jtv")
  for (s,df) in stats
    open("$s.tex", "w") do io
      latex_table(io, df, cols=cols, hdr_override=hdr_override)
    end
    df[:evals] = sum(df[c] for c in fieldnames(Counters))
  end
  cols = [:status, :objective, :elapsed_time, :evals]
  hdr_override = Dict(:objective => "\\(f(x)\\)", :elapsed_time => "\\(\\Delta t\\)", :status => "S")
  df_join = join(stats, cols, invariant_cols = [:name, :nvar, :ncon], hdr_override=hdr_override)
  open("SideBySide.tex", "w") do io
    latex_table(io, df_join)
  end

  cost_time(df) = (df.status .!= :first_order) * Inf + df.elapsed_time
  cost_eval(df) = (df.status .!= :first_order) * Inf + sum(df[c] for c in fieldnames(Counters))
  performance_profile(stats, cost_time)
  png("profile-time")
  performance_profile(stats, cost_eval)
  png("profile-eval")
end

#stats = runcutest()
profile_and_tables(stats)
