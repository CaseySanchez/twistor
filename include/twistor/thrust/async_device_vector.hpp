#pragma once

#include <thrust/device_vector.h>

#include "twistor/thrust/cuda/async_allocator.hpp"

THRUST_NAMESPACE_BEGIN

template <typename T>
using async_device_vector = thrust::device_vector<T, thrust::cuda::async_allocator<T>>;

THRUST_NAMESPACE_END