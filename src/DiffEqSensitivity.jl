__precompile__()

module DiffEqSensitivity

using DiffEqBase, Compat, ForwardDiff, DiffEqDiffTools, DiffEqCallbacks, QuadGK

abstract type SensitivityFunction end

include("derivative_wrappers.jl")
include("local_sensitivity.jl")
include("adjoint_sensitivity.jl")

export extract_local_sensitivities

export ODELocalSensitvityFunction, ODELocalSensitivityProblem, SensitivityFunction,
       ODEAdjointSensitivityProblem, ODEAdjointProblem, AdjointSensitivityIntegrand,
       adjoint_sensitivities
end # module