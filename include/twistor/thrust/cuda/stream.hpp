#pragma once

#include <thrust/detail/config.h>

#include <cuda_runtime_api.h>

namespace detail::thrust::cuda
{
    class stream
    {
        cudaStream_t m_stream;

    public:
        stream()
        {
            if (cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking) != cudaSuccess) {
                throw std::bad_alloc();
            }
        }

        ~stream()
        {
            cudaStreamDestroy(m_stream);
        }

        void synchronize()
        {
            cudaStreamSynchronize(m_stream);
        }

        operator cudaStream_t() const
        {
            return m_stream;
        }
    };
}

THRUST_NAMESPACE_BEGIN

namespace cuda
{
    using stream = ::detail::thrust::cuda::stream;
}

THRUST_NAMESPACE_END