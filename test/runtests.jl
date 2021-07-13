using DiffEqSensitivity, SafeTestsets
using Test, Pkg

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = Sys.iswindows() && haskey(ENV,"APPVEYOR")
const is_TRAVIS = haskey(ENV,"TRAVIS")

function activate_downstream_env()
    Pkg.activate("downstream")
    Pkg.develop(PackageSpec(path=dirname(@__DIR__)))
    Pkg.instantiate()
end

@time begin

if GROUP == "All" || GROUP == "Core1" || GROUP == "Downstream"
    @time @safetestset "Forward Sensitivity" begin include("forward.jl") end
    @time @safetestset "Adjoint Sensitivity" begin include("adjoint.jl") end
    @time @safetestset "Sparse Adjoint Sensitivity" begin include("sparse_adjoint.jl") end
    @time @safetestset "Second Order Sensitivity" begin include("second_order.jl") end
    @time @safetestset "Concrete Solve Derivatives" begin include("concrete_solve_derivatives.jl") end
    @time @safetestset "Branching Derivatives" begin include("branching_derivatives.jl") end
    @time @safetestset "Derivative Shapes" begin include("derivative_shapes.jl") end
    @time @safetestset "save_idxs" begin include("save_idxs.jl") end
    @time @safetestset "ArrayPartitions" begin include("array_partitions.jl") end
    @time @safetestset "Complex Adjoints" begin include("complex_adjoints.jl") end
    @time @safetestset "Forward Remake" begin include("forward_remake.jl") end
    @time @safetestset "DiscreteProblem Adjoints" begin include("discrete.jl") end
    @time @safetestset "Time Type Mixing Adjoints" begin include("time_type_mixing.jl") end
    @time @safetestset "GSA tests" begin include("gsa.jl") end
end

if GROUP == "All" || GROUP == "Core2"
    @time @safetestset "hasbranching" begin include("hasbranching.jl") end
    @time @safetestset "Literal Adjoint" begin include("literal_adjoint.jl") end
    @time @safetestset "ForwardDiff Chunking Adjoints" begin include("forward_chunking.jl") end
    @time @safetestset "Stiff Adjoints" begin include("stiff_adjoints.jl") end
    @time @safetestset "Autodiff Events" begin include("autodiff_events.jl") end
    @time @safetestset "Null Parameters" begin include("null_parameters.jl") end
    @time @safetestset "Forward Mode Prob Kwargs" begin include("forward_prob_kwargs.jl") end
    @time @safetestset "Steady State Adjoint" begin include("steady_state.jl") end
    @time @safetestset "Concrete Solve Derivatives of Second Order ODEs" begin include("second_order_odes.jl") end
end

if GROUP == "All" || GROUP == "Core3"
    @time @safetestset "Callbacks with Adjoints" begin include("callbacks.jl") end
    @time @safetestset "Shadowing Tests" begin include("shadowing.jl") end
end

if GROUP == "All" || GROUP == "SDE1"
    @time @safetestset "SDE Adjoint" begin include("sde_stratonovich.jl") end
    @time @safetestset "SDE Scalar Noise" begin include("sde_scalar_stratonovich.jl") end
    @time @safetestset "SDE Checkpointing" begin include("sde_checkpointing.jl") end
end

if GROUP == "All" || GROUP == "SDE2"
    @time @safetestset "SDE Non-Diagonal Noise" begin include("sde_nondiag_stratonovich.jl") end
end

if GROUP == "All" || GROUP == "SDE3"
    @time @safetestset "RODE Tests" begin include("rode.jl") end
    @time @safetestset "SDE Ito Conversion Tests" begin include("sde_transformation_test.jl") end
    @time @safetestset "SDE Ito Scalar Noise" begin include("sde_scalar_ito.jl") end
end

if GROUP == "DiffEqFlux"
    activate_downstream_env()
    @time @safetestset "Callback - ReverseDiff" begin include("downstream/callback_reversediff.jl") end
    @time @safetestset "ForwardDiff Sparsity Components" begin include("downstream/forwarddiffsensitivity_sparsity_components.jl") end
    @time @safetestset "SDE - Neural" begin include("downstream/sde_neural.jl") end
    @time @safetestset "Complex No u" begin include("downstream/complex_no_u.jl") end
end

if GROUP == "GPU"
    activate_downstream_env()
    @time @safetestset "Mixed GPU/CPU" begin include("gpu/mixed_gpu_cpu_adjoint.jl") end
end
end
