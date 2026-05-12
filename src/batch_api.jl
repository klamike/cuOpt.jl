function cuOptSolveBatchLP(
    problem,
    settings,
    batch_size,
    objective_coefficients,
    objective_coefficients_size,
    constraint_lower_bounds,
    constraint_lower_bounds_size,
    constraint_upper_bounds,
    constraint_upper_bounds_size,
    variable_lower_bounds,
    variable_lower_bounds_size,
    variable_upper_bounds,
    variable_upper_bounds_size,
    objective_offsets,
    objective_offsets_size,
    solution_ptr,
)
    return ccall(
        (:cuOptSolveBatchLP, libcuopt),
        cuopt_int_t,
        (
            cuOptOptimizationProblem,
            cuOptSolverSettings,
            cuopt_int_t,
            Ptr{cuopt_float_t},
            cuopt_int_t,
            Ptr{cuopt_float_t},
            cuopt_int_t,
            Ptr{cuopt_float_t},
            cuopt_int_t,
            Ptr{cuopt_float_t},
            cuopt_int_t,
            Ptr{cuopt_float_t},
            cuopt_int_t,
            Ptr{cuopt_float_t},
            cuopt_int_t,
            Ptr{cuOptSolution},
        ),
        problem,
        settings,
        batch_size,
        objective_coefficients,
        objective_coefficients_size,
        constraint_lower_bounds,
        constraint_lower_bounds_size,
        constraint_upper_bounds,
        constraint_upper_bounds_size,
        variable_lower_bounds,
        variable_lower_bounds_size,
        variable_upper_bounds,
        variable_upper_bounds_size,
        objective_offsets,
        objective_offsets_size,
        solution_ptr,
    )
end

function cuOptGetBatchSize(solution, batch_size_ptr)
    return ccall(
        (:cuOptGetBatchSize, libcuopt),
        cuopt_int_t,
        (cuOptSolution, Ptr{cuopt_int_t}),
        solution,
        batch_size_ptr,
    )
end

function cuOptGetBatchTerminationStatus(solution, batch_index, termination_status_ptr)
    return ccall(
        (:cuOptGetBatchTerminationStatus, libcuopt),
        cuopt_int_t,
        (cuOptSolution, cuopt_int_t, Ptr{cuopt_int_t}),
        solution,
        batch_index,
        termination_status_ptr,
    )
end

function cuOptGetBatchObjectiveValue(solution, batch_index, objective_value_ptr)
    return ccall(
        (:cuOptGetBatchObjectiveValue, libcuopt),
        cuopt_int_t,
        (cuOptSolution, cuopt_int_t, Ptr{cuopt_float_t}),
        solution,
        batch_index,
        objective_value_ptr,
    )
end

function cuOptGetBatchPrimalSolution(solution, batch_index, solution_values)
    return ccall(
        (:cuOptGetBatchPrimalSolution, libcuopt),
        cuopt_int_t,
        (cuOptSolution, cuopt_int_t, Ptr{cuopt_float_t}),
        solution,
        batch_index,
        solution_values,
    )
end

function cuOptGetBatchDualSolution(solution, batch_index, dual_solution)
    return ccall(
        (:cuOptGetBatchDualSolution, libcuopt),
        cuopt_int_t,
        (cuOptSolution, cuopt_int_t, Ptr{cuopt_float_t}),
        solution,
        batch_index,
        dual_solution,
    )
end
