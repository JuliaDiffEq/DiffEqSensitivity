
struct SteadyStateAdjointSensitivityFunction{C<:AdjointDiffCache,Alg<:SteadyStateAdjoint,uType,SType,CV,λType,VJPType} <: SensitivityFunction
  diffcache::C
  sensealg::Alg
  discrete::Bool
  y::uType
  sol::SType
  colorvec::CV
  λ::λType
  vjp::VJPType
#  linsolve::LS
end

function SteadyStateAdjointSensitivityFunction(g,sensealg,discrete,sol,dg,colorvec)
  @unpack f, p, u0 = sol.prob

  diffcache, y = adjointdiffcache(g,sensealg,discrete,sol,dg;quad=false)

  λ = zero(y) # solution of f_x^T λ = g_x^T, Eq. (2) in  https://math.mit.edu/~stevenj/18.336/adjoint.pdf
  #linsolve = sensealg.linsolve(Val{:init},diffcache.uf,y)
  vjp = similar(λ, length(p))
  SteadyStateAdjointSensitivityFunction(diffcache,sensealg,discrete,y,sol,colorvec,λ,vjp)
end

@noinline function SteadyStateAdjointProblem(sol,sensealg::SteadyStateAdjoint,g,dg)
  @unpack f, p = sol.prob

  discrete = false

  p === DiffEqBase.NullParameters() && error("Your model does not have parameters, and thus it is impossible to calculate the derivative of the solution with respect to the parameters. Your model must have parameters to use parameter sensitivity calculations!")

  sense = SteadyStateAdjointSensitivityFunction(g,sensealg,discrete,sol,dg,f.colorvec)
  @unpack diffcache, y, sol, λ, vjp = sense
  if DiffEqBase.has_jac(f)
     f.jac(diffcache.J,y,p,nothing)
  else
     if DiffEqBase.isinplace(sol.prob)
       jacobian!(diffcache.J, diffcache.uf, y, diffcache.f_cache, sensealg, diffcache.jac_config)
     else
        temp = jacobian(diffcache.uf, y, sensealg)
        @. diffcache.J = temp
     end
  end

  if dg != nothing
    if g!= nothing
      dg(vec(diffcache.dg_val),y,p,nothing,nothing)
    else
      @. diffcache.dg_val = dg
    end
  else
    if g != nothing
      gradient!(vec(diffcache.dg_val),diffcache.g,y,sensealg,diffcache.g_grad_config)
    end
  end

  λ .= diffcache.J'\vec(diffcache.dg_val') # use linsolve here
  vecjacobian!(vec(diffcache.dg_val), λ, p, nothing, sense, dgrad=vjp, dy=nothing)

  if g != nothing
    # compute del g/del p
    dg_dp_val = zero(p)
    dg_dp = ParamGradientWrapper(g,nothing,y)
    dg_dp_config = build_grad_config(sensealg,dg_dp,p,y)
    gradient!(dg_dp_val,dg_dp,p,sensealg,dg_dp_config)

    @. dg_dp_val = dg_dp_val - vjp
    return dg_dp_val

  else
    return -vjp
  end
end
