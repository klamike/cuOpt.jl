module cuOptBatchOptInterfaceExt

import BatchOptInterface as BOI
import BatchQuadraticModels as BQM
import MathOptInterface as MOI
import NLPModels
import cuOpt

const MOIU = MOI.Utilities
const VI = MOI.VariableIndex
const SAF = MOI.ScalarAffineFunction{Float64}
const SQF = MOI.ScalarQuadraticFunction{Float64}
const LinQuad = Union{VI,SAF,SQF}
const ScalarSets = Union{
    MOI.LessThan{Float64},
    MOI.GreaterThan{Float64},
    MOI.EqualTo{Float64},
    MOI.Interval{Float64},
}

mutable struct _BatchOptimizer <: MOI.AbstractOptimizer
    cache::MOIU.UniversalFallback{MOIU.Model{Float64}}
    options::Dict{String,Any}
    silent::Bool
    batch_model::Any
    info::Any
    stats::Any
    constraints::Any
end

function cuOpt.BatchOptimizer()
    return _BatchOptimizer(
        MOIU.UniversalFallback(MOIU.Model{Float64}()),
        Dict{String,Any}(),
        false,
        nothing,
        nothing,
        nothing,
        nothing,
    )
end

MOI.get(::_BatchOptimizer, ::MOI.SolverName) = "cuOpt-Batch-PDLP"
MOI.supports_incremental_interface(::_BatchOptimizer) = true
MOI.is_empty(model::_BatchOptimizer) = MOI.is_empty(model.cache)

function MOI.empty!(model::_BatchOptimizer)
    MOI.empty!(model.cache)
    model.batch_model = nothing
    model.info = nothing
    model.stats = nothing
    model.constraints = nothing
    return
end

MOI.copy_to(dest::_BatchOptimizer, src::MOI.ModelLike) = MOIU.default_copy_to(dest, src)

function _invalidate!(model::_BatchOptimizer)
    model.batch_model = nothing
    model.info = nothing
    model.stats = nothing
    model.constraints = nothing
    return
end

function MOI.add_variable(model::_BatchOptimizer)
    _invalidate!(model)
    return MOI.add_variable(model.cache)
end

function MOI.add_variables(model::_BatchOptimizer, n::Int)
    _invalidate!(model)
    return MOI.add_variables(model.cache, n)
end

function MOI.add_constraint(model::_BatchOptimizer, func::MOI.AbstractFunction, set::MOI.AbstractSet)
    _invalidate!(model)
    return MOI.add_constraint(model.cache, func, set)
end

function MOI.delete(model::_BatchOptimizer, index::MOI.Index)
    _invalidate!(model)
    MOI.delete(model.cache, index)
    return
end

function MOI.modify(
    model::_BatchOptimizer,
    ci::MOI.ConstraintIndex,
    change::MOI.AbstractFunctionModification,
)
    _invalidate!(model)
    MOI.modify(model.cache, ci, change)
    return
end

function MOI.set(
    model::_BatchOptimizer,
    attr::Union{MOI.AbstractOptimizerAttribute,MOI.AbstractModelAttribute},
    value,
)
    _invalidate!(model)
    if attr isa MOI.RawOptimizerAttribute
        model.options[attr.name] = value
    else
        MOI.set(model.cache, attr, value)
    end
    return
end

function MOI.set(model::_BatchOptimizer, ::MOI.Silent, value::Bool)
    model.silent = value
    return
end

function MOI.set(model::_BatchOptimizer, attr::MOI.AbstractVariableAttribute, vi::VI, value)
    _invalidate!(model)
    MOI.set(model.cache, attr, vi, value)
    return
end

function MOI.set(
    model::_BatchOptimizer,
    attr::MOI.AbstractConstraintAttribute,
    ci::MOI.ConstraintIndex,
    value,
)
    _invalidate!(model)
    MOI.set(model.cache, attr, ci, value)
    return
end

function MOI.get(
    model::_BatchOptimizer,
    attr::Union{MOI.AbstractOptimizerAttribute,MOI.AbstractModelAttribute},
)
    if attr isa MOI.RawOptimizerAttribute
        return model.options[attr.name]
    elseif attr isa MOI.Silent
        return model.silent
    end
    return MOI.get(model.cache, attr)
end

MOI.get(model::_BatchOptimizer, attr::MOI.AbstractVariableAttribute, vi::VI) =
    MOI.get(model.cache, attr, vi)

MOI.get(model::_BatchOptimizer, attr::MOI.AbstractConstraintAttribute, ci::MOI.ConstraintIndex) =
    MOI.get(model.cache, attr, ci)

function MOI.supports(
    ::_BatchOptimizer,
    attr::Union{
        MOI.ObjectiveSense,
        MOI.ObjectiveFunction{<:LinQuad},
        MOI.RawOptimizerAttribute,
        MOI.Silent,
    },
)
    return true
end

function MOI.supports(
    model::_BatchOptimizer,
    attr::Union{MOI.AbstractOptimizerAttribute,MOI.AbstractModelAttribute},
)
    return MOI.supports(model.cache, attr)
end

MOI.supports(::_BatchOptimizer, ::MOI.VariablePrimalStart, ::Type{VI}) = true

function MOI.supports(model::_BatchOptimizer, attr::MOI.AbstractVariableAttribute, ::Type{VI})
    return MOI.supports(model.cache, attr, VI)
end

function MOI.supports(
    model::_BatchOptimizer,
    attr::MOI.AbstractConstraintAttribute,
    index_type::Type{<:MOI.ConstraintIndex},
)
    return MOI.supports(model.cache, attr, index_type)
end

_supported_set(::Type{<:ScalarSets}) = true
_supported_set(::Type{<:MOI.Parameter}) = true
_supported_set(::Type{<:BOI.Batched{S}}) where {S<:MOI.AbstractScalarSet} =
    _supported_set(S)
_supported_set(::Type{<:MOI.AbstractSet}) = false

function MOI.supports_add_constrained_variable(
    ::_BatchOptimizer,
    S::Type{<:MOI.AbstractScalarSet},
)
    return _supported_set(S)
end

function MOI.supports_constraint(::_BatchOptimizer, ::Type{VI}, S::Type{<:MOI.AbstractSet})
    return _supported_set(S)
end

function MOI.supports_constraint(
    ::_BatchOptimizer,
    F::Type{<:Union{SAF,SQF}},
    S::Type{<:MOI.AbstractSet},
)
    return _supported_set(S) && !(S <: MOI.Parameter)
end

MOI.is_valid(model::_BatchOptimizer, index::MOI.Index) = MOI.is_valid(model.cache, index)

function MOI.optimize!(model::_BatchOptimizer)
    batch_model, info = BQM.batch_qp_model(model.cache)
    if !(batch_model isa BQM.BatchLinearModel)
        throw(MOI.UnsupportedAttribute(MOI.ObjectiveFunction{SQF}()))
    end
    options = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in model.options)
    options[:log_to_console] = model.silent ? 0 : get(options, :log_to_console, 1)
    stats = BQM.Solvers.cuopt(batch_model; options...)
    constraints = Matrix{Float64}(undef, batch_model.meta.ncon, batch_model.meta.nbatch)
    NLPModels.cons!(batch_model, stats.solution, constraints)
    model.batch_model = batch_model
    model.info = info
    model.stats = stats
    model.constraints = constraints
    return
end

const TERMINATION_STATUS = Dict{Int32,MOI.TerminationStatusCode}(
    cuOpt.CUOPT_TERMINATION_STATUS_NO_TERMINATION => MOI.OPTIMIZE_NOT_CALLED,
    cuOpt.CUOPT_TERMINATION_STATUS_OPTIMAL => MOI.OPTIMAL,
    cuOpt.CUOPT_TERMINATION_STATUS_INFEASIBLE => MOI.INFEASIBLE,
    cuOpt.CUOPT_TERMINATION_STATUS_UNBOUNDED => MOI.DUAL_INFEASIBLE,
    cuOpt.CUOPT_TERMINATION_STATUS_ITERATION_LIMIT => MOI.ITERATION_LIMIT,
    cuOpt.CUOPT_TERMINATION_STATUS_TIME_LIMIT => MOI.TIME_LIMIT,
    cuOpt.CUOPT_TERMINATION_STATUS_NUMERICAL_ERROR => MOI.NUMERICAL_ERROR,
    cuOpt.CUOPT_TERMINATION_STATUS_PRIMAL_FEASIBLE => MOI.ALMOST_OPTIMAL,
    cuOpt.CUOPT_TERMINATION_STATUS_FEASIBLE_FOUND => MOI.ALMOST_OPTIMAL,
    cuOpt.CUOPT_TERMINATION_STATUS_CONCURRENT_LIMIT => MOI.OTHER_LIMIT,
    cuOpt.CUOPT_TERMINATION_STATUS_WORK_LIMIT => MOI.OTHER_LIMIT,
    cuOpt.CUOPT_TERMINATION_STATUS_UNBOUNDED_OR_INFEASIBLE => MOI.INFEASIBLE_OR_UNBOUNDED,
)

function _status(model::_BatchOptimizer, result_index::Int)
    if model.stats === nothing
        return MOI.OPTIMIZE_NOT_CALLED
    end
    return TERMINATION_STATUS[model.stats.raw_status[result_index]]
end

MOI.get(model::_BatchOptimizer, ::MOI.ResultCount) =
    model.stats === nothing ? 0 : length(model.stats.raw_status)

function MOI.get(model::_BatchOptimizer, ::MOI.TerminationStatus)
    n = MOI.get(model, MOI.ResultCount())
    if iszero(n)
        return MOI.OPTIMIZE_NOT_CALLED
    end
    statuses = [_status(model, i) for i in 1:n]
    return all(==(first(statuses)), statuses) ? first(statuses) : MOI.OTHER_ERROR
end

function MOI.get(model::_BatchOptimizer, attr::BOI.BatchTerminationStatus)
    MOI.check_result_index_bounds(model, attr)
    return _status(model, attr.result_index)
end

function MOI.get(model::_BatchOptimizer, attr::MOI.PrimalStatus)
    if attr.result_index > MOI.get(model, MOI.ResultCount())
        return MOI.NO_SOLUTION
    end
    status = _status(model, attr.result_index)
    return status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL) ? MOI.FEASIBLE_POINT : MOI.NO_SOLUTION
end

function MOI.get(model::_BatchOptimizer, attr::MOI.DualStatus)
    if attr.result_index > MOI.get(model, MOI.ResultCount())
        return MOI.NO_SOLUTION
    end
    status = _status(model, attr.result_index)
    return status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL) ? MOI.FEASIBLE_POINT : MOI.NO_SOLUTION
end

function MOI.get(model::_BatchOptimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    return model.stats.objective[attr.result_index]
end

function MOI.get(model::_BatchOptimizer, ::MOI.RawStatusString)
    if MOI.get(model, MOI.ResultCount()) == 0
        return "OPTIMIZE_NOT_CALLED"
    end
    return string(model.stats.raw_status)
end

function MOI.get(model::_BatchOptimizer, attr::MOI.VariablePrimal, vi::VI)
    MOI.check_result_index_bounds(model, attr)
    parameters = model.info.parameter_values[attr.result_index]
    if haskey(parameters, vi)
        return parameters[vi]
    end
    mapped = model.info.index_maps[attr.result_index][vi]
    return model.stats.solution[mapped.value, attr.result_index]
end

function MOI.get(
    model::_BatchOptimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex,
)
    MOI.check_result_index_bounds(model, attr)
    if ci isa MOI.ConstraintIndex{VI,<:Union{MOI.Parameter,BOI.Batched{<:MOI.Parameter}}}
        vi = MOI.get(model.cache, MOI.ConstraintFunction(), ci)
        return MOI.get(model, MOI.VariablePrimal(attr.result_index), vi)
    end
    mapped = model.info.index_maps[attr.result_index][ci]
    if mapped isa MOI.ConstraintIndex{VI}
        return model.stats.solution[mapped.value, attr.result_index]
    end
    return model.constraints[mapped.value, attr.result_index]
end

function MOI.get(
    model::_BatchOptimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{VI},
)
    MOI.check_result_index_bounds(model, attr)
    throw(MOI.GetAttributeNotAllowed(attr, "cuOpt batch LP variable-bound duals are not exposed."))
end

function MOI.get(
    model::_BatchOptimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex,
)
    MOI.check_result_index_bounds(model, attr)
    mapped = model.info.index_maps[attr.result_index][ci]
    return model.stats.multipliers[mapped.value, attr.result_index]
end

end # module
