#pragma once

#include <gatl/gacsta.hpp>

using namespace gacsta;

using vector_t = decltype(full_kvector_t<float, 6, 1>());
using motor_t = decltype(full_kvector_t<float, 6, 0>() + full_kvector_t<float, 6, 2>() + full_kvector_t<float, 6, 4>() + full_kvector_t<float, 6, 6>());