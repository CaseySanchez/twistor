/**
 * @file computation_info.hpp
 * @author Casey Sanchez
 * @brief Enumeration for computation status information.
 */

#pragma once

enum class ComputationInfo
{
    Success,
    NumericalIssue,
    NoConvergence,
    InvalidInput
};