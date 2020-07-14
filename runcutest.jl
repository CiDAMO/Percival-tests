using CUTEst, NLPModels, NLPModelsIpopt, Percival, Plots, SolverBenchmark, SolverTools
plotlyjs(size=(600,400))

function runcutest()
  pnames = sort(CUTEst.select(min_con=1,
                              max_var=10,
                              max_con=10))
  problems = (CUTEstModel(p) for p in pnames)

  ipopt_wrapper(nlp; kwargs...) = ipopt(nlp,
                                        max_cpu_time=30.0,
                                        constr_viol_tol=1e-8,
                                        print_level=0;
                                        kwargs...)

  solvers = Dict(:Percival => percival, :Ipopt => ipopt_wrapper)

  stats = bmark_solvers(solvers, problems)
end

function profile_and_tables(stats)
  #cols = [:name, :nvar, :ncon, :status, :objective, :dual_feas, :primal_feas, :elapsed_time, :neval_obj, :neval_grad, :neval_cons, :neval_jac, :neval_jprod, :neval_jtprod, :neval_hess, :neval_hprod]
  #hdr_override = Dict(:objective => "\\(f(x)\\)", :dual_feas => "dual", :primal_feas => "prim", :elapsed_time => "\\(\\Delta t\\)",
                      #:neval_obj => "\\#f", :neval_grad => "\\#g", :neval_hess => "\\#H", :neval_hprod => "\\#Hv",
                      #:neval_cons => "\\#c", :neval_jac => "\\#J", :neval_jprod => "\\#Jv", :neval_jtprod => "\\#Jtv")
  cols = [:name, :nvar, :ncon, :status, :objective, :dual_feas, :primal_feas, :elapsed_time, :evals, :success]
  hdr_override = Dict(:objective => "\\(f(x)\\)")

  for (s,df) in stats
    df[:feas] = (df[:primal_feas]) .<= 1e-8
  end

  fmin = min.(stats[:Percival].objective + .!stats[:Percival].feas*1e20,
              stats[:Ipopt].objective + .!stats[:Ipopt].feas*1e20)

  for (s,df) in stats
    ineq = abs.(df[:objective] - fmin)./(max.(1.0 ,abs.(fmin))) .<= 1e-4

    df[:success] = min.(ineq, df[:feas])*1
    df[:evals] = df[:neval_grad] - (df[:neval_obj] - df[:neval_grad])/3
    open("$s.tex", "w") do io
      latex_table(io, df, cols=cols, hdr_override=hdr_override)
    end
  end

  #cols = [:status, :objective, :elapsed_time, :evals]
  #hdr_override = Dict(:objective => "\\(f(x)\\)", :elapsed_time => "\\(\\Delta t\\)", :status => "S")
  #df_join = join(stats, cols, invariant_cols = [:name, :nvar, :ncon], hdr_override=hdr_override)
  #open("SideBySide.tex", "w") do io
    #latex_table(io, df_join)
  #end
  cost_time(df) = (df.success .!= true) * Inf + df.elapsed_time
  cost_eval(df) = (df.success .!= true) * Inf + df.evals
  performance_profile(stats, cost_time)
  png("profile-time")
  performance_profile(stats, cost_eval)
  png("profile-eval")

  cost_status_time(df) = (df.status .!= :first_order) * Inf + df.elapsed_time
  cost_status_eval(df) = (df.status .!= :first_order) * Inf + df.evals
  performance_profile(stats, cost_status_time)
  png("profile-status-time")
  performance_profile(stats, cost_status_eval)
  png("profile-status-eval")
end

stats = runcutest()
profile_and_tables(stats)
