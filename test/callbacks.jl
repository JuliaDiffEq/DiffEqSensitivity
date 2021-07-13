using StochasticDiffEq, OrdinaryDiffEq, Zygote
using DiffEqSensitivity, Test, ForwardDiff

abstol=1e-12
reltol=1e-12
savingtimes=0.5

function test_discrete_callback(cb, tstops, g, dg!, cboop=nothing)
  function fiip(du,u,p,t)
    du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
    du[2] = dy = -p[3]*u[2] + p[4]*u[1]*u[2]
  end
  function foop(u,p,t)
    dx = p[1]*u[1] - p[2]*u[1]*u[2]
    dy = -p[3]*u[2] + p[4]*u[1]*u[2]
    [dx,dy]
  end

  p = [1.5,1.0,3.0,1.0]; u0 = [1.0;1.0]

  prob = ODEProblem(fiip,u0,(0.0,10.0),p)
  proboop = ODEProblem(foop,u0,(0.0,10.0),p)

  sol1 = solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=tstops,abstol=abstol,reltol=reltol,saveat=savingtimes)
  sol2 = solve(prob,Tsit5(),u0=u0,p=p,tstops=tstops,abstol=abstol,reltol=reltol,saveat=savingtimes)

  if cb.save_positions == [1,1]
    @test length(sol1.t) != length(sol2.t)
  else
    @test length(sol1.t) == length(sol2.t)
  end

  du01,dp1 = Zygote.gradient(
    (u0,p)->g(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=tstops,abstol=abstol,reltol=reltol,saveat=savingtimes,sensealg=BacksolveAdjoint())),
    u0,p)

  du01b,dp1b = Zygote.gradient(
    (u0,p)->g(solve(proboop,Tsit5(),u0=u0,p=p,callback=cb,tstops=tstops,abstol=abstol,reltol=reltol,saveat=savingtimes,sensealg=BacksolveAdjoint())),
    u0,p)

  du01c,dp1c = Zygote.gradient(
    (u0,p)->g(solve(proboop,Tsit5(),u0=u0,p=p,callback=cb,tstops=tstops,abstol=abstol,reltol=reltol,saveat=savingtimes,sensealg=BacksolveAdjoint(checkpointing=false))),
    u0,p)

  if cboop === nothing
    du02,dp2 = Zygote.gradient(
    (u0,p)->g(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=tstops,abstol=abstol,reltol=reltol,saveat=savingtimes,sensealg=ReverseDiffAdjoint()))
    ,u0,p)
  else
    du02,dp2 = Zygote.gradient(
      (u0,p)->g(solve(prob,Tsit5(),u0=u0,p=p,callback=cboop,tstops=tstops,abstol=abstol,reltol=reltol,saveat=savingtimes,sensealg=ReverseDiffAdjoint()))
      ,u0,p)
  end

  du03,dp3 = Zygote.gradient(
    (u0,p)->g(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=tstops,abstol=abstol,reltol=reltol,saveat=savingtimes,sensealg=InterpolatingAdjoint(checkpointing=true))),
    u0,p)

  du03c,dp3c = Zygote.gradient(
    (u0,p)->g(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=tstops,abstol=abstol,reltol=reltol,saveat=savingtimes,sensealg=InterpolatingAdjoint(checkpointing=false))),
    u0,p)

  du04,dp4 = Zygote.gradient(
    (u0,p)->g(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,tstops=tstops,abstol=abstol,reltol=reltol,saveat=savingtimes,sensealg=QuadratureAdjoint())),
    u0,p)

  dstuff = ForwardDiff.gradient(
    (θ)->g(solve(prob,Tsit5(),u0=θ[1:2],p=θ[3:6],callback=cb,tstops=tstops,abstol=abstol,reltol=reltol,saveat=savingtimes)),
    [u0;p])

  @info dstuff

  @test du01 ≈ dstuff[1:2]
  @test dp1 ≈ dstuff[3:6]
  @test du01b ≈ dstuff[1:2]
  @test dp1b ≈ dstuff[3:6]
  @test du01c ≈ dstuff[1:2]
  @test dp1c ≈ dstuff[3:6]
  @test du01 ≈ du02
  @test du01 ≈ du03 rtol=1e-7
  @test du01 ≈ du03c rtol=1e-7
  @test du03 ≈ du03c
  @test du01 ≈ du04
  @test dp1 ≈ dp2
  @test dp1 ≈ dp3
  @test dp1 ≈ dp3c
  @test dp1 ≈ dp4 rtol=1e-7

  @test du02 ≈ dstuff[1:2]
  @test dp2 ≈ dstuff[3:6]

  cb2 = DiffEqSensitivity.track_callbacks(CallbackSet(cb),prob.tspan[1],prob.u0,prob.p,BacksolveAdjoint())
  sol_track = solve(prob,Tsit5(),u0=u0,p=p,callback=cb2,tstops=tstops,abstol=abstol,reltol=reltol,saveat=savingtimes)
  #cb_adj = DiffEqSensitivity.setup_reverse_callbacks(cb2,BacksolveAdjoint())

  adj_prob = ODEAdjointProblem(sol_track,BacksolveAdjoint(),dg!,sol_track.t,nothing,
  						 callback = cb2,
  						 abstol=abstol,reltol=reltol)
  adj_sol = solve(adj_prob, Tsit5(), abstol=abstol,reltol=reltol)
  @test du01 ≈ -adj_sol[1:2,end]
  @test dp1 ≈ adj_sol[3:6,end]


  # adj_prob = ODEAdjointProblem(sol_track,InterpolatingAdjoint(),dg!,sol_track.t,nothing,
  # 						 callback = cb2,
  # 						 abstol=abstol,reltol=reltol)
  # adj_sol = solve(adj_prob, Tsit5(), abstol=abstol,reltol=reltol)
  #
  # @test du01 ≈ -adj_sol[1:2,end]
  # @test dp1 ≈ adj_sol[3:6,end]
end

function test_continuous_wrt_discrete_callback()
  # test the continuous callbacks wrt to the equivalent discrete callback
  function f(du,u,p,t)
    #Bouncing Ball
    du[1] = u[2]
    du[2] = -p[1]
  end

  # no saving in Callbacks; prescribed vafter and vbefore; loss on the endpoint

  tstop = 3.1943828249997
  vbefore = -31.30495168499705
  vafter = 25.04396134799764

  u0 = [50.0,0.0]
  tspan = (0.0,5.0)
  p = [9.8, 0.8]

  prob = ODEProblem(f,u0,tspan,p)

  function condition(u,t,integrator) # Event when event_f(u,t) == 0
     t - tstop
  end
  function affect!(integrator)
    integrator.u[2] += vafter-vbefore
  end
  cb = ContinuousCallback(condition,affect!,save_positions=(false,false))


  condition2(u,t,integrator) = t == tstop
  cb2 = DiscreteCallback(condition2,affect!,save_positions=(false,false))


  du01,dp1 = Zygote.gradient(
    (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,
     callback=cb2,tstops=[tstop],
     sensealg=BacksolveAdjoint(),
     saveat=tspan[2], save_start=false)),u0,p)

  du02,dp2 = Zygote.gradient(
    (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,
     callback=cb,
     sensealg=BacksolveAdjoint(),
     saveat=tspan[2], save_start=false)),u0,p)

  dstuff = ForwardDiff.gradient((θ)-> sum(solve(prob,Tsit5(),u0=θ[1:2],p=θ[3:4],
     	callback=cb,saveat=tspan[2], save_start=false)),[u0;p])

  @info dstuff
  @test du01 ≈ dstuff[1:2]
  @test dp1 ≈ dstuff[3:4]
  @test du02 ≈ dstuff[1:2]
  @test dp2 ≈ dstuff[3:4]

  # no saving in Callbacks; prescribed vafter and vbefore; loss on the endpoint by slicing
  du01,dp1 = Zygote.gradient(
    (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,
     callback=cb2,tstops=[tstop],
     sensealg=BacksolveAdjoint())[end]),u0,p)

  du02,dp2 = Zygote.gradient(
    (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,
     callback=cb,
     sensealg=BacksolveAdjoint())[end]),u0,p)

  dstuff = ForwardDiff.gradient((θ)-> sum(solve(prob,Tsit5(),u0=θ[1:2],p=θ[3:4],
     	callback=cb)[end]),[u0;p])

  @info dstuff
  @test du01 ≈ dstuff[1:2]
  @test dp1 ≈ dstuff[3:4]
  @test du02 ≈ dstuff[1:2]
  @test dp2 ≈ dstuff[3:4]

  # with saving in Callbacks; prescribed vafter and vbefore; loss on the endpoint
  cb = ContinuousCallback(condition,affect!,save_positions=(true,true))
  cb2 = DiscreteCallback(condition2,affect!,save_positions=(true,true))

  du01,dp1 = Zygote.gradient(
    (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,
     callback=cb2,tstops=[tstop],
     sensealg=BacksolveAdjoint(),
     saveat=tspan[2], save_start=false)),u0,p)

  du02,dp2 = Zygote.gradient(
    (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,
     callback=cb,
     sensealg=BacksolveAdjoint(),
     saveat=tspan[2], save_start=false)),u0,p)

  dstuff = ForwardDiff.gradient((θ)-> sum(solve(prob,Tsit5(),u0=θ[1:2],p=θ[3:4],
     	callback=cb,saveat=tspan[2], save_start=false)),[u0;p])

  @info dstuff
  @test du01 ≈ dstuff[1:2]
  @test dp1 ≈ dstuff[3:4]
  @test du02 ≈ dstuff[1:2]
  @test dp2 ≈ dstuff[3:4]

  # with saving in Callbacks; prescribed vafter and vbefore; loss on the endpoint by slicing
  du01,dp1 = Zygote.gradient(
    (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,
     callback=cb2,tstops=[tstop],
     sensealg=BacksolveAdjoint())[end]),u0,p)

  du02,dp2 = Zygote.gradient(
    (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,
     callback=cb,
     sensealg=BacksolveAdjoint())[end]),u0,p)

  dstuff = ForwardDiff.gradient((θ)-> sum(solve(prob,Tsit5(),u0=θ[1:2],p=θ[3:4],
     	callback=cb)[end]),[u0;p])

  @info dstuff
  @test du01 ≈ dstuff[1:2]
  @test dp1 ≈ dstuff[3:4]
  @test du02 ≈ dstuff[1:2]
  @test dp2 ≈ dstuff[3:4]

  # with saving in Callbacks;  different affect function
  function affect2!(integrator)
    integrator.u[2] = -integrator.p[2]*integrator.u[2]
  end
  cb = ContinuousCallback(condition,affect2!,save_positions=(true,true))

  cb2 = DiscreteCallback(condition2,affect2!,save_positions=(true,true))

  du01,dp1 = Zygote.gradient(
    (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,
     callback=cb2,tstops=[tstop],
     sensealg=BacksolveAdjoint(),
     saveat=tspan[2], save_start=false)),u0,p)

  du02,dp2 = Zygote.gradient(
    (u0,p)->sum(solve(prob,Tsit5(),u0=u0,p=p,
     callback=cb,
     sensealg=BacksolveAdjoint(),
     saveat=tspan[2], save_start=false)),u0,p)

  dstuff = ForwardDiff.gradient((θ)-> sum(solve(prob,Tsit5(),u0=θ[1:2],p=θ[3:4],
     	callback=cb,saveat=tspan[2], save_start=false)),[u0;p])

  @info dstuff
  @test du01 ≈ dstuff[1:2]
  @test dp1 ≈ dstuff[3:4]
  @test du02 ≈ dstuff[1:2]
  @test dp2 ≈ dstuff[3:4]
  @test du01 ≈ du02
  @test dp1 ≈ dp2
end


function test_continuous_callback(cb, g, dg!)
  function fiip(du,u,p,t)
    du[1] = u[2]
    du[2] = -p[1]
  end
  function foop(u,p,t)
    dx = u[2]
    dy = -p[1]
    [dx,dy]
  end

  u0 = [50.0,0.0]
  tspan = (0.0,5.0)
  p = [9.8, 0.8]

  prob = ODEProblem(fiip,u0,tspan,p)
  proboop = ODEProblem(fiip,u0,tspan,p)

  sol1 = solve(prob,Tsit5(),u0=u0,p=p,callback=cb,abstol=abstol,reltol=reltol,saveat=savingtimes)
  sol2 = solve(prob,Tsit5(),u0=u0,p=p,abstol=abstol,reltol=reltol,saveat=savingtimes)

  if cb.save_positions == [1,1]
    @test length(sol1.t) != length(sol2.t)
  else
    @test length(sol1.t) == length(sol2.t)
  end

  du01,dp1 = @time Zygote.gradient(
    (u0,p)->g(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,abstol=abstol,reltol=reltol,saveat=savingtimes,sensealg=BacksolveAdjoint())),
    u0,p)

  du01b,dp1b = Zygote.gradient(
    (u0,p)->g(solve(proboop,Tsit5(),u0=u0,p=p,callback=cb,abstol=abstol,reltol=reltol,saveat=savingtimes,sensealg=BacksolveAdjoint())),
    u0,p)

  du01c,dp1c = Zygote.gradient(
    (u0,p)->g(solve(proboop,Tsit5(),u0=u0,p=p,callback=cb,abstol=abstol,reltol=reltol,saveat=savingtimes,sensealg=BacksolveAdjoint(checkpointing=false))),
    u0,p)

  @test_broken du02,dp2 = @time Zygote.gradient(
    (u0,p)->g(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,abstol=abstol,reltol=reltol,saveat=savingtimes,sensealg=ReverseDiffAdjoint()))
    ,u0,p)

  du03,dp3 = @time Zygote.gradient(
    (u0,p)->g(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,abstol=abstol,reltol=reltol,saveat=savingtimes,sensealg=InterpolatingAdjoint(checkpointing=true))),
    u0,p)

  du03c,dp3c = Zygote.gradient(
    (u0,p)->g(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,abstol=abstol,reltol=reltol,saveat=savingtimes,sensealg=InterpolatingAdjoint(checkpointing=false))),
    u0,p)

  du04,dp4 = @time Zygote.gradient(
    (u0,p)->g(solve(prob,Tsit5(),u0=u0,p=p,callback=cb,abstol=abstol,reltol=reltol,saveat=savingtimes,sensealg=QuadratureAdjoint())),
    u0,p)

  dstuff = @time ForwardDiff.gradient(
    (θ)->g(solve(prob,Tsit5(),u0=θ[1:2],p=θ[3:4],callback=cb,abstol=abstol,reltol=reltol,saveat=savingtimes)),
    [u0;p])

  @info dstuff

  @test du01 ≈ dstuff[1:2]
  @test dp1 ≈ dstuff[3:4]
  @test du01b ≈ dstuff[1:2]
  @test dp1b ≈ dstuff[3:4]
  @test du01c ≈ dstuff[1:2]
  @test dp1c ≈ dstuff[3:4]
  @test_broken du01 ≈ du02
  @test du01 ≈ du03 rtol=1e-7
  @test du01 ≈ du03c rtol=1e-7
  @test du03 ≈ du03c
  @test du01 ≈ du04
  @test_broken dp1 ≈ dp2
  @test dp1 ≈ dp3
  @test dp1 ≈ dp3c
  @test dp1 ≈ dp4 rtol=1e-7

  @test_broken du02 ≈ dstuff[1:2]
  @test_broken dp2 ≈ dstuff[3:4]

  cb2 = DiffEqSensitivity.track_callbacks(CallbackSet(cb),prob.tspan[1],prob.u0,prob.p,BacksolveAdjoint())
  sol_track = solve(prob,Tsit5(),u0=u0,p=p,callback=cb2,abstol=abstol,reltol=reltol,saveat=savingtimes)

  adj_prob = ODEAdjointProblem(sol_track,BacksolveAdjoint(),dg!,sol_track.t,nothing,
  						 callback = cb2,
  						 abstol=abstol,reltol=reltol)
  adj_sol = solve(adj_prob, Tsit5(), abstol=abstol,reltol=reltol)
  @test du01 ≈ -adj_sol[1:2,end]
  @test dp1 ≈ adj_sol[3:4,end]


  # adj_prob = ODEAdjointProblem(sol_track,InterpolatingAdjoint(),dg!,sol_track.t,nothing,
  # 						 callback = cb2,
  # 						 abstol=abstol,reltol=reltol)
  # adj_sol = solve(adj_prob, Tsit5(), abstol=abstol,reltol=reltol)
  #
  # @test du01 ≈ -adj_sol[1:2,end]
  # @test dp1 ≈ adj_sol[3:6,end]
end

function test_SDE_callbacks()
  function dt!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α*x - β*x*y
    du[2] = dy = -δ*y + γ*x*y
  end

  function dW!(du, u, p, t)
    du[1] = 0.1u[1]
    du[2] = 0.1u[2]
  end

  u0 = [1.0,1.0]
  tspan = (0.0, 10.0)
  p = [2.2, 1.0, 2.0, 0.4]
  prob_sde = SDEProblem(dt!, dW!, u0, tspan,p)

  condition(u,t,integrator) = integrator.t>9.0 #some condition
  function affect!(integrator)
  	 #println("Callback")  #some callback
  end
  cb = DiscreteCallback(condition,affect!,save_positions=(false,false))

  function predict_sde(p)
    return Array(solve(prob_sde, EM(), p=p, saveat=savingtimes,sensealg = ForwardDiffSensitivity(), dt=0.001, callback=cb))
  end

  loss_sde(p)= sum(abs2, x-1 for x in predict_sde(p))

  loss_sde(p)
  @time dp = gradient(p) do p
  	loss_sde(p)
  end

  @test !iszero(dp[1])
end

@testset "Test callbacks" begin
  println("Discrete Callbacks")
  @testset "Discrete callbacks" begin
  	@testset "ODEs" begin
  	  println("ODEs")
      @testset "simple loss function" begin
        g(sol) = sum(sol)
        function dg!(out,u,p,t,i)
          (out.=-1)
        end
        @testset "callbacks with no effect" begin
          condition(u,t,integrator) = t == 5
          affect!(integrator) = integrator.u[1] += 0.0
          cb = DiscreteCallback(condition,affect!,save_positions=(false,false))
          tstops=[5.0]
          test_discrete_callback(cb,tstops,g,dg!)
        end
        @testset "callbacks with no effect except saving the state" begin
          condition(u,t,integrator) = t == 5
          affect!(integrator) = integrator.u[1] += 0.0
          cb = DiscreteCallback(condition,affect!)
          tstops=[5.0]
          test_discrete_callback(cb,tstops,g,dg!)
        end
        @testset "callback at single time point" begin
          condition(u,t,integrator) = t == 5
          affect!(integrator) = integrator.u[1] += 2.0
          cb = DiscreteCallback(condition,affect!)
          tstops=[5.0]
          test_discrete_callback(cb,tstops,g,dg!)
        end
        @testset "callback at multiple time points" begin
          affecttimes = [2.03,4.0,8.0]
          condition(u,t,integrator) = t ∈ affecttimes
          affect!(integrator) = integrator.u[1] += 2.0
          cb = DiscreteCallback(condition,affect!)
          test_discrete_callback(cb,affecttimes,g,dg!)
        end
        @testset "state-dependent += callback at single time point" begin
          condition(u,t,integrator) = t == 5
          affect!(integrator) = (integrator.u .+= integrator.p[2]/8*sin.(integrator.u))
          cb = DiscreteCallback(condition,affect!)
          tstops=[5.0]
          test_discrete_callback(cb,tstops,g,dg!)
        end
        @testset "other callback at single time point" begin
          condition(u,t,integrator) = t == 5
          affect!(integrator) = (integrator.u[1] = 2.0; @show "triggered!")
          cb = DiscreteCallback(condition,affect!)
          tstops=[5.0]
          test_discrete_callback(cb,tstops,g,dg!)
        end
        @testset "parameter changing callback at single time point" begin
          condition(u,t,integrator) = t == 5.1
          affect!(integrator) = (integrator.p .= 2*integrator.p .- 0.5)
          affect(integrator) = (integrator.p = 2*integrator.p .- 0.5)
          cb = DiscreteCallback(condition,affect!)
          cboop = DiscreteCallback(condition,affect)
          cb = DiscreteCallback(condition,affect!)
          tstops=[5.1]
          test_discrete_callback(cb,tstops,g,dg!,cboop)
        end
      end
      @testset "MSE loss function" begin
        g(u) = sum((1.0.-u).^2)./2
        dg!(out,u,p,t,i) = (out.=1.0.-u)
        @testset "callbacks with no effect" begin
          condition(u,t,integrator) = t == 5
          affect!(integrator) = integrator.u[1] += 0.0
          cb = DiscreteCallback(condition,affect!,save_positions=(false,false))
          tstops=[5.0]
          test_discrete_callback(cb,tstops,g,dg!)
        end
        @testset "callbacks with no effect except saving the state" begin
          condition(u,t,integrator) = t == 5
          affect!(integrator) = integrator.u[1] += 0.0
          cb = DiscreteCallback(condition,affect!)
          tstops=[5.0]
          test_discrete_callback(cb,tstops,g,dg!)
        end
        @testset "callback at single time point" begin
          condition(u,t,integrator) = t == 5
          affect!(integrator) = integrator.u[1] += 2.0
          cb = DiscreteCallback(condition,affect!)
          tstops=[5.0]
          test_discrete_callback(cb,tstops,g,dg!)
        end
        @testset "callback at multiple time points" begin
          affecttimes = [2.03,4.0,8.0]
          condition(u,t,integrator) = t ∈ affecttimes
          affect!(integrator) = integrator.u[1] += 2.0
          cb = DiscreteCallback(condition,affect!)
          test_discrete_callback(cb,affecttimes,g,dg!)
        end
        @testset "state-dependent += callback at single time point" begin
          condition(u,t,integrator) = t == 5
          affect!(integrator) = (integrator.u .+= integrator.p[2]/8*sin.(integrator.u))
          cb = DiscreteCallback(condition,affect!)
          tstops=[5.0]
          test_discrete_callback(cb,tstops,g,dg!)
        end
        @testset "other callback at single time point" begin
          condition(u,t,integrator) = t == 5
          affect!(integrator) = (integrator.u[1] = 2.0; @show "triggered!")
          cb = DiscreteCallback(condition,affect!)
          tstops=[5.0]
          test_discrete_callback(cb,tstops,g,dg!)
        end
        @testset "parameter changing callback at single time point" begin
          condition(u,t,integrator) = t == 5.1
          affect!(integrator) = (integrator.p .= 2*integrator.p .- 0.5)
          affect(integrator) = (integrator.p = 2*integrator.p .- 0.5)
          cb = DiscreteCallback(condition,affect!)
          cboop = DiscreteCallback(condition,affect)
          tstops=[5.1]
          test_discrete_callback(cb,tstops,g,dg!,cboop)
        end
      end
  	end
  	@testset "SDEs" begin
  	  println("SDEs")
      test_SDE_callbacks()
  	end
  end

  println("Continuous Callbacks")
  @testset "Continuous callbacks" begin
    @testset "Compare with respect to discrete callback" begin
      test_continuous_wrt_discrete_callback()
    end
    @testset "simple loss function" begin
      g(sol) = sum(sol)
      function dg!(out,u,p,t,i)
        (out.=-1)
      end
      @testset "callbacks with no effect" begin
        condition(u,t,integrator) = u[1] # Event when event_f(u,t) == 0
        affect!(integrator) = (integrator.u[2] += 0)
        cb = ContinuousCallback(condition,affect!,save_positions=(false,false))
        test_continuous_callback(cb,g,dg!)
      end
      @testset "callbacks with no effect except saving the state" begin
        condition(u,t,integrator) = u[1]
        affect!(integrator) = (integrator.u[2] += 0)
        cb = ContinuousCallback(condition,affect!)
        test_continuous_callback(cb,g,dg!)
      end
      @testset "+= callback" begin
        condition(u,t,integrator) = u[1]
        affect!(integrator) = (integrator.u[2] += 50.0)
        cb = ContinuousCallback(condition,affect!)
        test_continuous_callback(cb,g,dg!)
      end
      @testset "= callback" begin
        condition(u,t,integrator) = u[1]
        affect!(integrator) = (integrator.u[2] = -integrator.p[2]*integrator.u[2])
        cb = ContinuousCallback(condition,affect!)
        test_continuous_callback(cb,g,dg!)
      end
    end
  end
end
