using BlockDiagonals

struct ODEForwardSensitivityFunction{iip,F,A,Tt,J,JP,S,PJ,TW,TWt,UF,PF,JC,PJC,Alg,fc,JM,pJM,MM,CV} <: DiffEqBase.AbstractODEFunction{iip}
  f::F
  analytic::A
  tgrad::Tt
  jac::J
  jac_prototype::JP
  sparsity::S
  paramjac::PJ
  Wfact::TW
  Wfact_t::TWt
  uf::UF
  pf::PF
  J::JM
  pJ::pJM
  jac_config::JC
  paramjac_config::PJC
  alg::Alg
  numparams::Int
  numindvar::Int
  f_cache::fc
  mass_matrix::MM
  isautojacvec::Bool
  colorvec::CV
end

struct NILSSForwardSensitivityFunction{iip,sensefunType,senseType,MM} <: DiffEqBase.AbstractODEFunction{iip}
  S::sensefunType
  sensealg::senseType
  nus::Int
  mass_matrix::MM
end

function ODEForwardSensitivityFunction(f,analytic,tgrad,jac,jac_prototype,sparsity,paramjac,Wfact,Wfact_t,uf,pf,u0,
                                    jac_config,paramjac_config,alg,p,f_cache,mm,
                                    isautojacvec,colorvec,nus)
  numparams = length(p)
  numindvar = length(u0)
  J = isautojacvec ? nothing : Matrix{eltype(u0)}(undef,numindvar,numindvar)
  pJ = Matrix{eltype(u0)}(undef,numindvar,numparams) # number of funcs size

  sensefun = ODEForwardSensitivityFunction{isinplace(f),typeof(f),typeof(analytic),
                             typeof(tgrad),typeof(jac),typeof(jac_prototype),typeof(sparsity),
                             typeof(paramjac),
                             typeof(Wfact),typeof(Wfact_t),typeof(uf),
                             typeof(pf),typeof(jac_config),
                             typeof(paramjac_config),typeof(alg),
                             typeof(f_cache),
                             typeof(J),typeof(pJ),typeof(mm),typeof(f.colorvec)}(
                             f,analytic,tgrad,jac,jac_prototype,sparsity,paramjac,Wfact,Wfact_t,uf,pf,J,pJ,
                             jac_config,paramjac_config,alg,
                             numparams,numindvar,f_cache,mm,isautojacvec,colorvec)
  if nus!==nothing
    sensefun = NILSSForwardSensitivityFunction{isinplace(f), typeof(sensefun),
                             typeof(alg),typeof(mm)}(sensefun,alg,nus,mm)
  end

  return sensefun
end

function sensefun2odefun(sense::ODEForwardSensitivityFunction, u0)
  @unpack jac_prototype, jac, sparsity, uf, J, jac_config, alg, f_cache, mass_matrix, colorvec = sense
  jac_prototype === nothing && (jac_prototype = u0 .* u0' .* false)
  jac_prototype = ForwardSensitivityJacobian(jac_prototype)
  mass_matrix = mass_matrix === I ? I : BlockDiagonal(mass_matrix, Diagonal(I, size(J, 1)))
  colorvec !== nothing && (colorvec = [colorvec; colorvec])
  newjac = if DiffEqBase.has_jac(sense)
    let jac=sense.jac, n=length(u0)
      (J, u, p, t) -> begin
        @assert J isa DiffEqBase.ForwardSensitivityJacobian
        J = parent(J)
        @views jac(J, u[1:n], p, t)
      end
    end
  else
    nothing
  end
  ODEFunction(sense; jac=newjac, jac_prototype=jac_prototype, colorvec=colorvec, mass_matrix=mass_matrix)
end

function (S::ODEForwardSensitivityFunction)(du,u,p,t)
  y = @view u[1:S.numindvar] # These are the independent variables
  dy = @view du[1:S.numindvar]
  S.f(dy,y,p,t) # Make the first part be the ODE

  # Now do sensitivities
  # Compute the Jacobian

  if !S.isautojacvec
    if DiffEqBase.has_jac(S.f)
      S.jac(S.J,y,p,t) # Calculate the Jacobian into J
    else
      S.uf.t = t
      jacobian!(S.J, S.uf, y, S.f_cache, S.alg, S.jac_config)
    end
  end


  if DiffEqBase.has_paramjac(S.f)
    S.paramjac(S.pJ,y,p,t) # Calculate the parameter Jacobian into pJ
  else
    S.pf.t = t
    S.pf.u .= y
    jacobian!(S.pJ, S.pf, p, S.f_cache, S.alg, S.paramjac_config)
  end


  # Compute the parameter derivatives
  for i in eachindex(p)
    Sj = @view u[i*S.numindvar+1:(i+1)*S.numindvar]
    dp = @view du[i*S.numindvar+1:(i+1)*S.numindvar]
    if !S.isautojacvec
      mul!(dp,S.J,Sj)
    else
      jacobianvec!(dp, S.uf, y, Sj, S.alg, S.jac_config)
    end
    dp .+= @view S.pJ[:,i]
  end
end

@deprecate ODELocalSensitivityProblem(args...;kwargs...) ODEForwardSensitivityProblem(args...;kwargs...)

struct ODEForwardSensitivityProblem{iip,A}
  sensealg::A
end

function ODEForwardSensitivityProblem(f,args...;kwargs...)
  ODEForwardSensitivityProblem(ODEFunction(f),args...;kwargs...)
end

function ODEForwardSensitivityProblem(prob::ODEProblem,alg;kwargs...)
  ODEForwardSensitivityProblem(prob.f,prob.u0,prob.tspan,prob.p,alg;kwargs...)
end

function ODEForwardSensitivityProblem(f::DiffEqBase.AbstractODEFunction,u0,
                                    tspan,p=nothing,
                                    alg::ForwardSensitivity = ForwardSensitivity();
                                    nus=nothing, # determine if Nilss is used
                                    w0=nothing,
                                    v0=nothing,
                                    kwargs...)
  isinplace = DiffEqBase.isinplace(f)
  # if there is an analytical Jacobian provided, we are not going to do automatic `jac*vec`
  isautojacvec = get_jacvec(alg)
  p === nothing && error("You must have parameters to use parameter sensitivity calculations!")
  uf = DiffEqBase.UJacobianWrapper(f,tspan[1],p)
  pf = DiffEqBase.ParamJacobianWrapper(f,tspan[1],copy(u0))
  if isautojacvec
    if alg_autodiff(alg)
      # if we are using automatic `jac*vec`, then we need to use a `jac_config`
      # that is a tuple in the form of `(seed, buffer)`
      jac_config_seed = ForwardDiff.Dual{typeof(jacobianvec!)}.(u0,u0)
      jac_config_buffer = similar(jac_config_seed)
      jac_config = jac_config_seed, jac_config_buffer
    else
      jac_config = (similar(u0),similar(u0))
    end
  elseif DiffEqBase.has_jac(f)
    jac_config = nothing
  else
    jac_config = build_jac_config(alg,uf,u0)
  end

  if DiffEqBase.has_paramjac(f)
    paramjac_config = nothing
  else
    paramjac_config = build_param_jac_config(alg,pf,u0,p)
  end

  # TODO: make it better
  if f.mass_matrix isa UniformScaling
    mm = f.mass_matrix
  else
    nn = size(f.mass_matrix, 1)
    mm = similar(f.mass_matrix, 2nn, 2nn)
    mm[1:nn, 1:nn] = f.mass_matrix
    mm[nn+1:2nn, nn+1:2nn] = f.mass_matrix
  end

  # TODO: Use user tgrad. iW can be safely ignored here.
  sense = ODEForwardSensitivityFunction(f,f.analytic,nothing,f.jac,
                                     f.jac_prototype,f.sparsity,f.paramjac,
                                     nothing,nothing,
                                     uf,pf,u0,jac_config,
                                     paramjac_config,alg,
                                     p,similar(u0),mm,
                                     isautojacvec,f.colorvec,nus)
  if nus===nothing
    sense_u0 = [u0;zeros(eltype(u0),sense.numindvar*sense.numparams)]
  else
    if w0===nothing && v0===nothing
      sense_u0 = [u0;zeros(eltype(u0),(nus+1)*sense.S.numindvar*sense.S.numparams)]
    else
      sense_u0 = [u0;w0;v0]
    end
  end
  ODEProblem(sensefun2odefun(sense,u0),sense_u0,tspan,p,
             ODEForwardSensitivityProblem{DiffEqBase.isinplace(f),
                                          typeof(alg)}(alg);
             kwargs...)
end

function seed_duals(x::AbstractArray{V},f,
                    ::ForwardDiff.Chunk{N} = ForwardDiff.Chunk(x,typemax(Int64)),
                    ) where {V,T,N}
  seeds = ForwardDiff.construct_seeds(ForwardDiff.Partials{N,V})
  duals = ForwardDiff.Dual{typeof(ForwardDiff.Tag(f,eltype(vec(x))))}.(vec(x),seeds)
end

has_continuous_callback(cb::DiscreteCallback) = false
has_continuous_callback(cb::ContinuousCallback) = true
has_continuous_callback(cb::CallbackSet) = !isempty(cb.continuous_callbacks)

function ODEForwardSensitivityProblem(f::DiffEqBase.AbstractODEFunction,u0,
                                      tspan,p,alg::ForwardDiffSensitivity;
                                      kwargs...)
  pdual = seed_duals(p,f)
  u0dual = convert.(eltype(pdual),u0)

  if (convert_tspan(alg) === nothing &&
    haskey(kwargs,:callback) && has_continuous_callback(kwargs.callback)
    ) || (convert_tspan(alg) !== nothing && convert_tspan(alg))
    tspandual = convert.(eltype(pdual),tspan)
  else
    tspandual = tspan
  end

  prob_dual = ODEProblem(f,u0dual,tspan,pdual,
                         ODEForwardSensitivityProblem{DiffEqBase.isinplace(f),
                                                      typeof(alg)}(alg);
                         kwargs...)
end

extract_local_sensitivities(sol, asmatrix::Val=Val(false)) = extract_local_sensitivities(sol,sol.prob.problem_type.sensealg, asmatrix)
extract_local_sensitivities(sol, asmatrix::Bool) = extract_local_sensitivities(sol, Val{asmatrix}())
extract_local_sensitivities(sol, i::Integer, asmatrix::Val=Val(false)) = _extract(sol, sol.prob.problem_type.sensealg, sol[i], asmatrix)
extract_local_sensitivities(sol, i::Integer, asmatrix::Bool) = extract_local_sensitivities(sol, i, Val{asmatrix}())
extract_local_sensitivities(sol, t::Union{Number,AbstractVector}, asmatrix::Val=Val(false)) = _extract(sol, sol.prob.problem_type.sensealg, sol(t), asmatrix)
extract_local_sensitivities(sol, t, asmatrix::Bool) = extract_local_sensitivities(sol, t, Val{asmatrix}())
extract_local_sensitivities(tmp, sol, t::Union{Number,AbstractVector}, asmatrix::Val=Val(false)) = _extract(sol, sol.prob.problem_type.sensealg, sol(tmp, t), asmatrix)
extract_local_sensitivities(tmp, sol, t, asmatrix::Bool) = extract_local_sensitivities(tmp, sol, t, Val{asmatrix}())

# Get ODE u vector and sensitivity values from all time points
function extract_local_sensitivities(sol,::ForwardSensitivity, ::Val{false})
  ni = sol.prob.f.f.numindvar
  u = sol[1:ni, :]
  du = [sol[ni*j+1:ni*(j+1),:] for j in 1:sol.prob.f.f.numparams]
  return u, du
end

function extract_local_sensitivities(sol,::ForwardDiffSensitivity, ::Val{false})
  u = ForwardDiff.value.(sol)
  du_full = ForwardDiff.partials.(sol)
  firststate = first(du_full)
  firstparam = first(firststate)
  Js = map(1:length(firstparam)) do j
    map(CartesianIndices(du_full)) do II
      du_full[II][j]
    end
  end
  return u, Js
end

function extract_local_sensitivities(sol,::ForwardSensitivity, ::Val{true})
  prob = sol.prob
  ni = prob.f.f.numindvar
  pn = prob.f.f.numparams
  jsize = (ni, pn)
  sol[1:ni, :], map(sol.u) do u
    collect(reshape((@view u[ni+1:end]), jsize))
  end
end

function extract_local_sensitivities(sol,::ForwardDiffSensitivity, ::Val{true})
  retu = ForwardDiff.value.(sol)
  jsize = length(sol.u[1]), ForwardDiff.npartials(sol.u[1][1])
  du = map(sol.u) do u
    du_i = similar(retu, jsize)
    for i in eachindex(u)
      du_i[i, :] = ForwardDiff.partials(u[i])
    end
    du_i
  end
  retu, du
end

# Get ODE u vector and sensitivity values from sensitivity problem u vector
function _extract(sol, sensealg::ForwardSensitivity, su::AbstractVector, asmatrix::Val = Val(false))
  u = view(su, 1:sol.prob.f.f.numindvar)
  du = _extract_du(sol, sensealg, su, asmatrix)
  return u, du
end

function _extract(sol, sensealg::ForwardDiffSensitivity, su::AbstractVector, asmatrix::Val = Val(false))
  u = ForwardDiff.value.(su)
  du = _extract_du(sol, sensealg, su, asmatrix)
  return u, du
end

# Get sensitivity values from sensitivity problem u vector (nested form)
function _extract_du(sol, ::ForwardSensitivity, su::Vector, ::Val{false})
  ni = sol.prob.f.f.numindvar
  return [view(su, ni*j+1:ni*(j+1)) for j in 1:sol.prob.f.f.numparams]
end

function _extract_du(sol, ::ForwardDiffSensitivity, su::Vector, ::Val{false})
  du_full = ForwardDiff.partials.(su)
  return [[du_full[i][j] for i in 1:size(du_full,1)] for j in 1:length(du_full[1])]
end

# Get sensitivity values from sensitivity problem u vector (matrix form)
function _extract_du(sol, ::ForwardSensitivity, su::Vector, ::Val{true})
  ni = sol.prob.f.f.numindvar
  np = sol.prob.f.f.numparams
  return view(reshape(su, ni, np+1), :, 2:np+1)
end

function _extract_du(sol, ::ForwardDiffSensitivity, su::Vector, ::Val{true})
  du_full = ForwardDiff.partials.(su)
  return [du_full[i][j] for i in 1:size(du_full,1), j in 1:length(du_full[1])]
end


### Bonus Pieces
function SciMLBase.remake(prob::ODEProblem{uType,tType,isinplace,P,F,K,<:ODEForwardSensitivityProblem};
                          f=nothing,tspan=nothing,u0=nothing,p=nothing,kwargs...) where
                          {uType,tType,isinplace,P,F,K}
    _p     = p     === nothing ? prob.p : p
    if f === nothing
      _f = prob.f.f.f
    elseif f isa ODEFunction && f.f isa ODEForwardSensitivityFunction
      _f = f.f.f
    else
      _f = f
    end
    _u0    = u0    === nothing ? prob.u0[1:prob.f.f.numindvar] : u0[1:prob.f.f.numindvar]
    _tspan = tspan === nothing ? prob.tspan : tspan
    ODEForwardSensitivityProblem(_f,_u0,
                                 _tspan,_p,ForwardSensitivity();
                                 prob.kwargs...,kwargs...)
end
