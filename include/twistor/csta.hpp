/**
 * @file csta.hpp
 * @brief Uses the `gacsta` namespace and provides type definitions.
 */

#pragma once

#include <type_traits>

#include <gatl/gacsta.hpp>

using namespace gacsta;

using scalar_t = decltype(
    full_kvector_t<float, 6, 0>()
);

using vector_t = decltype(
    full_kvector_t<float, 6, 1>()
);

using bivector_t = decltype(
    full_kvector_t<float, 6, 2>()
);

using trivector_t = decltype(
    full_kvector_t<float, 6, 3>()
);

using quadvector_t = decltype(
    full_kvector_t<float, 6, 4>()
);

using pentavector_t = decltype(
    full_kvector_t<float, 6, 5>()
);

using pseudoscalar_t = decltype(
    full_kvector_t<float, 6, 6>()
);

using motor_t = decltype(
    full_kvector_t<float, 6, 0>() +
    full_kvector_t<float, 6, 2>() +
    full_kvector_t<float, 6, 4>() +
    full_kvector_t<float, 6, 6>()
);
