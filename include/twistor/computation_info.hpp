/**
 * @file computation_info.hpp
 * @brief Enumeration for computation status information.
 */

#pragma once

namespace Twistor
{
    enum class ComputationInfo
    {
        Success,
        NumericalIssue,
        NoConvergence,
        InvalidInput
    };
}