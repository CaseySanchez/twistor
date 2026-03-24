#pragma once

#include <thrust/detail/config.h>

#include <cuda_runtime_api.h>

namespace detail::thrust::cuda
{
    template <typename T>
    class async_allocator
    {
        cudaStream_t m_stream;

    public:
        using value_type = T;
        using pointer = ::thrust::device_ptr<T>;
        using const_pointer = ::thrust::device_ptr<T const>;
        using reference = ::thrust::device_reference<T>;
        using const_reference = ::thrust::device_reference<T const>;
        using size_type = std::size_t;
        using difference_type = typename pointer::difference_type;

        async_allocator(cudaStream_t stream = 0) :
            m_stream { stream }
        {
        }

        pointer allocate(std::size_t size)
        {
            if (size == 0) {
                return nullptr;
            }

            void *ptr = nullptr;

            if (cudaMallocAsync(&ptr, size * sizeof(T), m_stream) != cudaSuccess) {
                throw std::bad_alloc();
            }

            return pointer(static_cast<T *>(ptr));
        }

        void deallocate(pointer ptr, [[maybe_unused]] std::size_t size) noexcept
        {
            if (ptr.get() == nullptr) {
                return;
            }

            cudaFreeAsync(ptr.get(), m_stream);
        }

        bool operator==(async_allocator const &other) const
        {
            return m_stream == other.m_stream;
        }

        bool operator!=(async_allocator const &other) const
        {
            return !(*this == other);
        }

        template <typename U>
        struct rebind
        {
            using other = async_allocator<U>;
        };
    };
}

THRUST_NAMESPACE_BEGIN

namespace cuda
{
    template <typename T>
    using async_allocator = ::detail::thrust::cuda::async_allocator<T>;
}

THRUST_NAMESPACE_END